import io
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image as PILImage
from langchain_core.documents import Document

from agent.context import BASE_ROLE_INSTRUCTIONS
from agent.document import DocumentIngestionJobRegistry, _build_child_docs_from_pages, load_document
from agent.memory import PDF_PAYLOAD_INDEX_FIELDS
from agent.ocr_engine import OcrResult
from agent.scanned_pdf_processor import (
    PageContent,
    classify_page_metrics,
    layout_result_to_page_contents,
    pages_to_markdown,
    process_pdf_pages,
)
from agent.tools import format_pdf_parent_record
from agent.vision_analyzer import LayoutRegion, PageLayoutResult, VisionAnalyzer
from lexical_index import PersistentLexicalIndex


class PdfIngestionTests(unittest.TestCase):
    def _create_text_pdf(self, path: Path):
        doc = fitz.open()
        page = doc.new_page(width=240, height=240)
        page.insert_text((24, 60), "Native text page with a result table.")
        doc.save(path)
        doc.close()

    def _create_scanned_pdf(self, path: Path):
        image = PILImage.new("RGB", (240, 240), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        doc = fitz.open()
        page = doc.new_page(width=240, height=240)
        page.insert_image(page.rect, stream=buffer.getvalue())
        doc.save(path)
        doc.close()

    def test_text_page_with_enough_text_is_text(self):
        self.assertEqual(
            classify_page_metrics(
                char_count=500,
                text_block_count=8,
                image_coverage_ratio=0.0,
                char_threshold=50,
            ),
            "text",
        )

    def test_low_text_page_with_large_image_is_scanned(self):
        self.assertEqual(
            classify_page_metrics(
                char_count=10,
                text_block_count=0,
                image_coverage_ratio=0.92,
                char_threshold=50,
            ),
            "scanned",
        )

    def test_short_caption_page_without_large_image_stays_text(self):
        self.assertEqual(
            classify_page_metrics(
                char_count=35,
                text_block_count=2,
                image_coverage_ratio=0.10,
                char_threshold=50,
            ),
            "text",
        )

    def test_pages_to_markdown_keeps_page_order_and_anchors(self):
        pages = [
            PageContent(
                page_number=2,
                page_type="scanned",
                text="Second page",
                content_source="ocr",
                ocr_engine="paddleocr",
                confidence=0.8,
                source_filename="paper.pdf",
            ),
            PageContent(
                page_number=1,
                page_type="text",
                text="First page",
                content_source="native_pdf_text",
                source_filename="paper.pdf",
            ),
        ]

        markdown = pages_to_markdown(pages)

        self.assertLess(markdown.index("<!-- page: 1"), markdown.index("<!-- page: 2"))
        self.assertIn("First page", markdown)
        self.assertIn("Second page", markdown)

    def test_page_content_from_ocr_sets_evidence_fields(self):
        ocr = OcrResult(page_number=4, text="OCR text", engine_used="tesseract", confidence=0.5)

        page = PageContent.from_ocr(
            page_number=4,
            source_filename="paper.pdf",
            ocr_result=ocr,
        )

        self.assertEqual(page.page_number, 4)
        self.assertEqual(page.page_type, "scanned")
        self.assertEqual(page.content_source, "ocr")
        self.assertEqual(page.ocr_engine, "tesseract")
        self.assertFalse(page.generated)
        self.assertEqual(page.metadata()["source"], "paper.pdf")

    def test_child_docs_inherit_page_metadata(self):
        pages = [
            PageContent(
                page_number=7,
                page_type="scanned",
                text="A scanned page about transformers and attention.",
                content_source="ocr",
                source_filename="paper.pdf",
                ocr_engine="paddleocr",
                confidence=0.77,
            )
        ]

        child_docs = _build_child_docs_from_pages("paper.pdf", pages)

        self.assertGreaterEqual(len(child_docs), 1)
        metadata = child_docs[0].metadata
        self.assertEqual(metadata["source"], "paper.pdf")
        self.assertEqual(metadata["page_number"], 7)
        self.assertEqual(metadata["page_type"], "scanned")
        self.assertEqual(metadata["content_source"], "ocr")
        self.assertEqual(metadata["ocr_engine"], "paddleocr")
        self.assertEqual(metadata["ocr_confidence"], 0.77)
        self.assertIn("parent_text", metadata)

    def test_formats_evidence_metadata_for_model(self):
        formatted = format_pdf_parent_record(
            {
                "source": "paper.pdf",
                "page_number": 12,
                "page_type": "scanned",
                "content_source": "ocr",
                "generated": False,
                "parent_text": "Evidence text",
            }
        )

        self.assertIn("来源文档：paper.pdf", formatted)
        self.assertIn("页码：12", formatted)
        self.assertIn("页面类型：scanned", formatted)
        self.assertIn("来源类型：ocr", formatted)
        self.assertIn("自动生成内容：否", formatted)
        self.assertTrue(formatted.endswith("Evidence text"))

    def test_pdf_evidence_fields_are_indexed(self):
        self.assertIn("page_type", PDF_PAYLOAD_INDEX_FIELDS)
        self.assertIn("content_source", PDF_PAYLOAD_INDEX_FIELDS)
        self.assertIn("evidence_type", PDF_PAYLOAD_INDEX_FIELDS)

    def test_context_mentions_ocr_and_generated_figure_evidence(self):
        self.assertIn("OCR", BASE_ROLE_INSTRUCTIONS)
        self.assertIn("自动图表描述", BASE_ROLE_INSTRUCTIONS)
        self.assertIn("不能当作论文原文引用", BASE_ROLE_INSTRUCTIONS)

    def test_process_pdf_pages_reuses_cached_ocr_result(self):
        class FakeOcr:
            def __init__(self, fail_if_called: bool = False):
                self.fail_if_called = fail_if_called
                self.calls = 0

            def extract_text(self, _image):
                self.calls += 1
                if self.fail_if_called:
                    raise AssertionError("OCR should not run when page cache exists")
                return OcrResult(page_number=0, text="Cached OCR text", engine_used="fake", confidence=0.91)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "scanned.pdf"
            cache_dir = tmp_path / "page-cache"
            self._create_scanned_pdf(pdf_path)

            first_ocr = FakeOcr()
            first_pages = process_pdf_pages(pdf_path, ocr_engine=first_ocr, cache_dir=cache_dir)

            second_ocr = FakeOcr(fail_if_called=True)
            second_pages = process_pdf_pages(pdf_path, ocr_engine=second_ocr, cache_dir=cache_dir)

        self.assertEqual(first_ocr.calls, 1)
        self.assertEqual(second_ocr.calls, 0)
        self.assertEqual([page.text for page in first_pages], ["Cached OCR text"])
        self.assertEqual([page.text for page in second_pages], ["Cached OCR text"])

    def test_layout_result_becomes_separate_table_and_generated_figure_pages(self):
        layout = PageLayoutResult(
            page_number=3,
            regions=[
                LayoutRegion(
                    region_type="table",
                    bbox=(0, 0, 100, 80),
                    confidence=0.88,
                    content="| Method | Score |\n| --- | --- |\n| A | 90 |",
                ),
                LayoutRegion(
                    region_type="figure",
                    bbox=(0, 90, 100, 180),
                    confidence=0.77,
                    content="图中展示了模型结构。",
                ),
            ],
        )

        pages = layout_result_to_page_contents(layout, source_filename="paper.pdf")

        self.assertEqual(len(pages), 2)
        table_page, figure_page = pages
        self.assertEqual(table_page.page_number, 3)
        self.assertEqual(table_page.content_source, "table_extraction")
        self.assertTrue(table_page.has_table)
        self.assertFalse(table_page.generated)
        self.assertIn("自动表格提取", table_page.text)
        self.assertEqual(figure_page.content_source, "vision_model")
        self.assertTrue(figure_page.has_figure)
        self.assertTrue(figure_page.generated)
        self.assertIn("自动图表描述", figure_page.text)

    def test_load_document_can_enable_visual_page_processing_for_text_pdf(self):
        class FakeStore:
            def __init__(self):
                self.docs = []

            def add_documents(self, docs):
                self.docs.extend(docs)

        class FakeVisionAnalyzer:
            def __init__(self):
                self.calls = 0

            def analyze_page(self, _image, page_number: int):
                self.calls += 1
                return PageLayoutResult(
                    page_number=page_number,
                    regions=[
                        LayoutRegion(
                            region_type="table",
                            bbox=(0, 0, 100, 80),
                            confidence=0.9,
                            content="| Metric | Value |\n| --- | --- |\n| F1 | 0.91 |",
                        )
                    ],
                )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "native.pdf"
            registry_path = tmp_path / "documents.json"
            self._create_text_pdf(pdf_path)
            store = FakeStore()
            analyzer = FakeVisionAnalyzer()

            with patch("agent.document.DOCS_FILE", registry_path):
                result = load_document(
                    str(pdf_path),
                    store,
                    user_id="tester",
                    include_visual_descriptions=True,
                    vision_analyzer=analyzer,
                )

        self.assertTrue(result["success"])
        self.assertEqual(analyzer.calls, 1)
        self.assertTrue(
            any(doc.metadata.get("content_source") == "table_extraction" for doc in store.docs)
        )

    def test_load_document_adds_page_metadata_for_native_text_pdf_by_default(self):
        class FakeStore:
            def __init__(self):
                self.docs = []

            def add_documents(self, docs):
                self.docs.extend(docs)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "native.pdf"
            registry_path = tmp_path / "documents.json"
            self._create_text_pdf(pdf_path)
            store = FakeStore()

            with patch("agent.document.DOCS_FILE", registry_path):
                result = load_document(str(pdf_path), store, user_id="tester")

        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(store.docs), 1)
        metadata = store.docs[0].metadata
        self.assertEqual(metadata["page_number"], 1)
        self.assertEqual(metadata["page_type"], "text")
        self.assertEqual(metadata["content_source"], "native_pdf_text")
        self.assertEqual(metadata["generated"], False)

    def test_document_ingestion_job_registry_tracks_background_result(self):
        registry = DocumentIngestionJobRegistry()

        job = registry.submit("load paper.pdf", lambda: {"success": True, "message": "loaded"})

        deadline = time.time() + 2
        status = registry.get(job["job_id"])
        while status["status"] not in {"succeeded", "failed"} and time.time() < deadline:
            time.sleep(0.01)
            status = registry.get(job["job_id"])

        self.assertEqual(status["status"], "succeeded")
        self.assertEqual(status["result"]["message"], "loaded")
        self.assertEqual(status["message"], "loaded")

    def test_load_document_reuses_registered_same_pdf_without_duplicate_indexing(self):
        class FakeStore:
            def __init__(self):
                self.calls = 0
                self.docs = []

            def add_documents(self, docs):
                self.calls += 1
                self.docs.extend(docs)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "native.pdf"
            registry_path = tmp_path / "documents.json"
            self._create_text_pdf(pdf_path)
            store = FakeStore()

            with patch("agent.document.DOCS_FILE", registry_path):
                first = load_document(str(pdf_path), store, user_id="tester")
                second = load_document(str(pdf_path), store, user_id="tester")

        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertTrue(second.get("reused"))
        self.assertEqual(store.calls, 1)

    def test_reindex_changed_pdf_cleans_old_vector_and_lexical_chunks(self):
        class FakeClient:
            def __init__(self):
                self.deleted = []

            def scroll(self, **_kwargs):
                return (
                    [
                        SimpleNamespace(id="old-point", payload={"metadata": {"source": "native.pdf"}}),
                        SimpleNamespace(id="other-point", payload={"metadata": {"source": "other.pdf"}}),
                    ],
                    None,
                )

            def delete(self, collection_name, points_selector):
                self.deleted.append((collection_name, points_selector))

        class FakeStore:
            def __init__(self):
                self.client = FakeClient()
                self.collection_name = "pdf_knowledge"
                self.docs = []

            def add_documents(self, docs):
                self.docs.extend(docs)

        class FakeLexicalIndex:
            def __init__(self):
                self.removed = []
                self.indexed = []

            def remove_source(self, user_id, source):
                self.removed.append((user_id, source))

            def index_documents(self, docs, user_id, source):
                self.indexed.append((len(docs), user_id, source))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "native.pdf"
            registry_path = tmp_path / "documents.json"
            self._create_text_pdf(pdf_path)
            registry_path.write_text(
                json.dumps(
                    {
                        "native.pdf": {
                            "filename": "native.pdf",
                            "title": "old",
                            "summary": "old",
                            "date_added": "2026-01-01 00:00",
                            "chunk_count": 1,
                            "file_hash": "old-hash",
                            "ingestion_profile": "default",
                        }
                    }
                ),
                encoding="utf-8",
            )
            store = FakeStore()
            lexical_index = FakeLexicalIndex()

            with patch("agent.document.DOCS_FILE", registry_path):
                result = load_document(
                    str(pdf_path),
                    store,
                    user_id="tester",
                    lexical_index=lexical_index,
                )

        self.assertTrue(result["success"])
        self.assertEqual(store.client.deleted, [("pdf_knowledge", ["old-point"])])
        self.assertEqual(lexical_index.removed, [("__shared_pdf__", "native.pdf")])
        self.assertEqual(lexical_index.indexed[0][1:], ("__shared_pdf__", "native.pdf"))

    def test_document_ingestion_job_registry_accepts_progress_updates(self):
        registry = DocumentIngestionJobRegistry()

        def worker(progress):
            progress("parsing", "Parsing PDF", 30)
            progress("indexing", "Indexing chunks", 80)
            return {"success": True, "message": "done"}

        job = registry.submit("load paper.pdf", worker)
        deadline = time.time() + 2
        status = registry.get(job["job_id"])
        while status["status"] not in {"succeeded", "failed"} and time.time() < deadline:
            time.sleep(0.01)
            status = registry.get(job["job_id"])

        self.assertEqual(status["status"], "succeeded")
        self.assertEqual(status["stage"], "completed")
        self.assertEqual(status["progress"], 100)
        self.assertEqual(status["message"], "done")

    def test_load_document_reports_progress_stages(self):
        class FakeStore:
            def __init__(self):
                self.docs = []

            def add_documents(self, docs):
                self.docs.extend(docs)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "native.pdf"
            registry_path = tmp_path / "documents.json"
            self._create_text_pdf(pdf_path)
            progress_events = []

            with patch("agent.document.DOCS_FILE", registry_path):
                result = load_document(
                    str(pdf_path),
                    FakeStore(),
                    user_id="tester",
                    progress_callback=lambda stage, message, progress: progress_events.append(
                        (stage, message, progress)
                    ),
                )

        self.assertTrue(result["success"])
        self.assertEqual(progress_events[0][0], "detecting")
        self.assertIn("parsing", [event[0] for event in progress_events])
        self.assertIn("indexing", [event[0] for event in progress_events])
        self.assertEqual(progress_events[-1][0], "completed")
        self.assertEqual(progress_events[-1][2], 100)

    def test_lexical_index_preserves_pdf_evidence_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = PersistentLexicalIndex(Path(tmp) / "lexical.db")
            doc = Document(
                page_content="adapter fusion result",
                metadata={
                    "chunk_id": "chunk-1",
                    "parent_id": "parent-1",
                    "parent_text": "adapter fusion result parent",
                    "source": "paper.pdf",
                    "page_number": 9,
                    "page_type": "scanned",
                    "content_source": "ocr",
                    "ocr_engine": "fake",
                    "ocr_confidence": 0.83,
                    "generated": False,
                },
            )

            index.index_documents([doc], user_id="__shared_pdf__", source="paper.pdf")
            results = index.search("adapter fusion", user_id="__shared_pdf__", source="paper.pdf", top_k=1)

        self.assertEqual(len(results), 1)
        metadata = results[0].metadata
        self.assertEqual(metadata["page_number"], 9)
        self.assertEqual(metadata["page_type"], "scanned")
        self.assertEqual(metadata["content_source"], "ocr")
        self.assertEqual(metadata["ocr_engine"], "fake")
        self.assertEqual(metadata["ocr_confidence"], 0.83)

    def test_vision_analyzer_cache_profile_includes_model_identity(self):
        class FakeVisionLlm:
            model_name = "glm-4v-test"

        profile = VisionAnalyzer(vision_llm=FakeVisionLlm()).cache_profile()

        self.assertIn("glm-4v-test", profile)
        self.assertIn("GLM4V_MAX_TOKENS", profile)


if __name__ == "__main__":
    unittest.main()
