"""扫描型/混合型 PDF 的页级处理编排器。"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from PIL import Image as PILImage

from agent.config import (
    OCR_FALLBACK_LANG,
    OCR_LANG,
    PDF_PAGE_CACHE_DIR,
    PDF_PAGE_CACHE_ENABLED,
    PDF_TO_IMAGE_DPI,
    SCANNED_PAGE_CHAR_THRESHOLD,
)
from agent.ocr_engine import OcrEngine, OcrResult

PDF_PAGE_CACHE_VERSION = "1"


@dataclass
class PageClassificationResult:
    """单页类型判断结果。"""

    page_number: int
    page_type: str
    char_count: int
    has_image: bool
    text_block_count: int = 0
    image_coverage_ratio: float = 0.0


@dataclass
class PdfClassificationResult:
    """整篇 PDF 的页面类型统计。"""

    total_pages: int
    text_pages: list[PageClassificationResult]
    scanned_pages: list[PageClassificationResult]

    @property
    def has_scanned_pages(self) -> bool:
        return bool(self.scanned_pages)

    @property
    def is_all_scanned(self) -> bool:
        return len(self.text_pages) == 0 and len(self.scanned_pages) > 0

    @property
    def is_mixed(self) -> bool:
        return len(self.text_pages) > 0 and len(self.scanned_pages) > 0

    @property
    def scanned_page_numbers(self) -> list[int]:
        return [page.page_number for page in self.scanned_pages]


@dataclass
class PageContent:
    """单页内容与证据来源 metadata。"""

    page_number: int
    page_type: str
    text: str
    content_source: str
    source_filename: str = ""
    ocr_engine: str = "none"
    confidence: float = 1.0
    generated: bool = False
    has_figure: bool = False
    has_table: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_native_text(cls, page_number: int, source_filename: str, text: str) -> "PageContent":
        return cls(
            page_number=page_number,
            page_type="text",
            text=text,
            content_source="native_pdf_text",
            source_filename=source_filename,
            ocr_engine="none",
            confidence=1.0,
            generated=False,
        )

    @classmethod
    def from_ocr(cls, page_number: int, source_filename: str, ocr_result: OcrResult) -> "PageContent":
        return cls(
            page_number=page_number,
            page_type="scanned",
            text=ocr_result.text,
            content_source="ocr",
            source_filename=source_filename,
            ocr_engine=ocr_result.engine_used,
            confidence=ocr_result.confidence,
            generated=False,
        )

    def metadata(self) -> dict:
        return {
            "source": self.source_filename,
            "page_number": self.page_number,
            "page_type": self.page_type,
            "content_source": self.content_source,
            "ocr_engine": self.ocr_engine,
            "ocr_confidence": self.confidence,
            "generated": self.generated,
            "has_figure": self.has_figure,
            "has_table": self.has_table,
            "evidence_type": self.content_source,
            **self.extra,
        }

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "page_type": self.page_type,
            "text": self.text,
            "content_source": self.content_source,
            "source_filename": self.source_filename,
            "ocr_engine": self.ocr_engine,
            "confidence": self.confidence,
            "generated": self.generated,
            "has_figure": self.has_figure,
            "has_table": self.has_table,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PageContent":
        return cls(
            page_number=int(payload["page_number"]),
            page_type=str(payload["page_type"]),
            text=str(payload.get("text", "")),
            content_source=str(payload["content_source"]),
            source_filename=str(payload.get("source_filename", "")),
            ocr_engine=str(payload.get("ocr_engine", "none")),
            confidence=float(payload.get("confidence", 1.0)),
            generated=bool(payload.get("generated", False)),
            has_figure=bool(payload.get("has_figure", False)),
            has_table=bool(payload.get("has_table", False)),
            extra=dict(payload.get("extra", {}) or {}),
        )


from agent.utils import file_sha256 as _file_sha256


def _ocr_cache_profile(ocr_engine: OcrEngine) -> str:
    profile = getattr(ocr_engine, "cache_profile", None)
    if callable(profile):
        return str(profile())
    return (
        f"{type(ocr_engine).__module__}.{type(ocr_engine).__qualname__};"
        f"ocr_lang={OCR_LANG};fallback={OCR_FALLBACK_LANG};dpi={PDF_TO_IMAGE_DPI};"
        f"threshold={SCANNED_PAGE_CHAR_THRESHOLD}"
    )


def _vision_cache_profile(vision_analyzer: object | None) -> str:
    if vision_analyzer is None:
        return "none"
    profile = getattr(vision_analyzer, "cache_profile", None)
    if callable(profile):
        return str(profile())
    return f"{type(vision_analyzer).__module__}.{type(vision_analyzer).__qualname__}"


def _page_cache_path(
    pdf_path: Path,
    ocr_engine: OcrEngine,
    cache_dir: Path | None = None,
    *,
    include_visual_descriptions: bool = False,
    vision_analyzer: object | None = None,
) -> Path:
    cache_root = cache_dir or PDF_PAGE_CACHE_DIR
    cache_key_payload = {
        "version": PDF_PAGE_CACHE_VERSION,
        "pdf_sha256": _file_sha256(pdf_path),
        "ocr_profile": _ocr_cache_profile(ocr_engine),
        "include_visual_descriptions": include_visual_descriptions,
        "vision_profile": _vision_cache_profile(vision_analyzer),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_root / f"{cache_key}.json"


def _load_cached_pages(cache_path: Path) -> list[PageContent] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("version") != PDF_PAGE_CACHE_VERSION:
            return None
        return [PageContent.from_dict(page) for page in payload.get("pages", [])]
    except Exception as exc:
        print(f"[PDFCache] 读取页级缓存失败，已忽略: {exc}")
        return None


def _save_cached_pages(cache_path: Path, pages: list[PageContent]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PDF_PAGE_CACHE_VERSION,
            "pages": [page.to_dict() for page in pages],
        }
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(cache_path)
    except Exception as exc:
        print(f"[PDFCache] 写入页级缓存失败，已忽略: {exc}")


def classify_page_metrics(
    char_count: int,
    text_block_count: int,
    image_coverage_ratio: float,
    char_threshold: int | None = None,
) -> str:
    """基于单页指标判断页面类型，便于测试和复用。"""
    threshold = char_threshold if char_threshold is not None else SCANNED_PAGE_CHAR_THRESHOLD
    if char_count >= threshold:
        return "text"
    if text_block_count == 0 and image_coverage_ratio >= 0.60:
        return "scanned"
    if char_count == 0 and image_coverage_ratio >= 0.35:
        return "scanned"
    return "text"


def _calculate_image_coverage_ratio(page) -> float:
    """估算页面被图片区域覆盖的比例。"""
    page_area = float(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0

    covered_area = 0.0
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        try:
            for rect in page.get_image_rects(xref):
                covered_area += float(rect.width * rect.height)
        except Exception:
            continue
    return min(covered_area / page_area, 1.0)


def _count_text_blocks(text_blocks: list) -> int:
    """统计 PyMuPDF text blocks 中真正的文本块数量。"""
    count = 0
    for block in text_blocks:
        if len(block) > 6 and block[6] != 0:
            continue
        if len(block) > 4 and str(block[4]).strip():
            count += 1
    return count


def classify_pages(pdf_path: str | Path, char_threshold: int | None = None) -> PdfClassificationResult:
    """逐页检测 PDF 页面类型。"""
    threshold = char_threshold if char_threshold is not None else SCANNED_PAGE_CHAR_THRESHOLD
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"无法打开 PDF 文件: {pdf_path}") from exc

    text_pages: list[PageClassificationResult] = []
    scanned_pages: list[PageClassificationResult] = []
    try:
        for page_index in range(len(doc)):
            page_number = page_index + 1
            try:
                page = doc[page_index]
                page_text = page.get_text()
                char_count = len(page_text.replace("\n", "").replace(" ", "").strip())
                text_block_count = _count_text_blocks(page.get_text("blocks") or [])
                image_list = page.get_images(full=True)
                has_image = bool(image_list)
                image_coverage_ratio = _calculate_image_coverage_ratio(page)
                page_type = classify_page_metrics(
                    char_count=char_count,
                    text_block_count=text_block_count,
                    image_coverage_ratio=image_coverage_ratio,
                    char_threshold=threshold,
                )
                page_result = PageClassificationResult(
                    page_number=page_number,
                    page_type=page_type,
                    char_count=char_count,
                    has_image=has_image,
                    text_block_count=text_block_count,
                    image_coverage_ratio=image_coverage_ratio,
                )
            except Exception as exc:
                print(f"[PageClassifier] 第 {page_number} 页解析异常，标记为扫描页: {exc}")
                page_result = PageClassificationResult(
                    page_number=page_number,
                    page_type="scanned",
                    char_count=0,
                    has_image=False,
                    text_block_count=0,
                    image_coverage_ratio=0.0,
                )

            if page_result.page_type == "text":
                text_pages.append(page_result)
            else:
                scanned_pages.append(page_result)
    finally:
        doc.close()

    return PdfClassificationResult(
        total_pages=len(text_pages) + len(scanned_pages),
        text_pages=text_pages,
        scanned_pages=scanned_pages,
    )


def render_page_image(page, dpi: int = PDF_TO_IMAGE_DPI) -> PILImage.Image:
    """把 PyMuPDF 页面渲染为 PIL 图片，避免额外依赖 poppler。"""
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return PILImage.open(io.BytesIO(pixmap.tobytes("png")))


def pages_to_markdown(pages: list[PageContent]) -> str:
    """按页码拼接 Markdown，并保留显式页锚点。"""
    chunks: list[str] = []
    # pages 列表在 process_pdf_pages 中按页码顺序构建，无需再排序
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        chunks.append(
            f"<!-- page: {page.page_number}; page_type: {page.page_type}; "
            f"content_source: {page.content_source} -->\n{text}"
        )
    return "\n\n".join(chunks)


def layout_result_to_page_contents(layout_result, source_filename: str) -> list[PageContent]:
    """把版面分析结果转换成独立证据块，避免混入 OCR 或原生 PDF 文本。"""
    page_number = int(getattr(layout_result, "page_number", 0) or 0)
    page_contents: list[PageContent] = []

    table_blocks: list[str] = []
    table_confidences: list[float] = []
    for index, region in enumerate(getattr(layout_result, "table_regions", []), start=1):
        content = str(getattr(region, "content", "")).strip()
        if not content:
            continue
        table_blocks.append(f"#### 表格 {index}\n{content}")
        table_confidences.append(float(getattr(region, "confidence", 0.0) or 0.0))
    if table_blocks:
        page_contents.append(
            PageContent(
                page_number=page_number,
                page_type="mixed",
                text="### 自动表格提取\n\n" + "\n\n".join(table_blocks),
                content_source="table_extraction",
                source_filename=source_filename,
                ocr_engine="none",
                confidence=max(table_confidences) if table_confidences else 0.0,
                generated=False,
                has_table=True,
            )
        )

    figure_blocks: list[str] = []
    figure_confidences: list[float] = []
    for index, region in enumerate(getattr(layout_result, "figure_regions", []), start=1):
        content = str(getattr(region, "content", "")).strip()
        if not content:
            continue
        figure_blocks.append(f"#### 图表 {index}\n{content}")
        figure_confidences.append(float(getattr(region, "confidence", 0.0) or 0.0))
    if figure_blocks:
        page_contents.append(
            PageContent(
                page_number=page_number,
                page_type="mixed",
                text="### 自动图表描述\n\n" + "\n\n".join(figure_blocks),
                content_source="vision_model",
                source_filename=source_filename,
                ocr_engine="none",
                confidence=max(figure_confidences) if figure_confidences else 0.0,
                generated=True,
                has_figure=True,
            )
        )

    return page_contents


def process_pdf_pages(
    pdf_path: str | Path,
    ocr_engine: OcrEngine | None = None,
    *,
    use_cache: bool = PDF_PAGE_CACHE_ENABLED,
    cache_dir: Path | None = None,
    vision_analyzer: object | None = None,
    include_visual_descriptions: bool = False,
) -> list[PageContent]:
    """把扫描型/混合型 PDF 处理为按页排序的 PageContent。"""
    path = Path(pdf_path)
    source_filename = path.name
    ocr = ocr_engine or OcrEngine()

    cache_path = (
        _page_cache_path(
            path,
            ocr,
            cache_dir,
            include_visual_descriptions=include_visual_descriptions,
            vision_analyzer=vision_analyzer,
        )
        if use_cache
        else None
    )
    if cache_path is not None:
        cached_pages = _load_cached_pages(cache_path)
        if cached_pages is not None:
            return cached_pages

    classification = classify_pages(path)
    scanned_pages = set(classification.scanned_page_numbers)
    page_contents: list[PageContent] = []

    doc = fitz.open(path)
    try:
        for page_index in range(len(doc)):
            page_number = page_index + 1
            page = doc[page_index]
            page_image = None
            if page_number in scanned_pages:
                page_image = render_page_image(page)
                ocr_result = ocr.extract_text(page_image)
                ocr_result.page_number = page_number
                page_contents.append(PageContent.from_ocr(page_number, source_filename, ocr_result))
            else:
                page_contents.append(
                    PageContent.from_native_text(
                        page_number=page_number,
                        source_filename=source_filename,
                        text=page.get_text().strip(),
                    )
                )

            if include_visual_descriptions and vision_analyzer is not None:
                if page_image is None:
                    page_image = render_page_image(page)
                layout_result = vision_analyzer.analyze_page(page_image, page_number=page_number)
                page_contents.extend(layout_result_to_page_contents(layout_result, source_filename))
    finally:
        doc.close()

    if cache_path is not None:
        _save_cached_pages(cache_path, page_contents)

    return page_contents
