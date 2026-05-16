# document.py
# PDF 文档的解析入库 + 元数据注册。
# 父子块策略：父块（1500字）保留完整上下文，子块（400字）用于精细检索。
# documents.json 持久化文档元数据，重启后不丢失。

import inspect
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import FieldCondition, Filter, MatchValue
from pydantic import BaseModel

from agent.error_recovery import ModelRecoveryManager
from agent.scanned_pdf_processor import PageContent, pages_to_markdown, process_pdf_pages
from agent.utils import extract_payload_metadata as _extract_payload_metadata, file_sha256 as _file_sha256
from agent.memory import PDF_COLLECTION_NAME
from lexical_index import PDF_SHARED_SCOPE, build_chunk_id

_BASE = Path(__file__).resolve().parent.parent  # 项目根目录，保证 documents.json 仍然落在原位置。
DOCS_FILE = _BASE / "documents.json"
REGISTRY_FALLBACK_SUMMARY = "（从现有知识库回填，未生成摘要）"  # 旧知识库回填 documents.json 时使用的默认摘要。
_DOCUMENT_REGISTRY_READY = False  # 每个进程只做一次文档注册回填，避免反复全库扫描。
_PARENT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
_CHILD_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


class DocumentIngestionJobRegistry:
    """进程内 PDF 入库任务表，用于 UI 展示后台处理状态。"""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, description: str, worker) -> dict:
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "description": description,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "入库任务已创建，等待执行。",
            "result": None,
            "error": "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": "",
            "finished_at": "",
        }
        with self._lock:
            self._jobs[job_id] = dict(job)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, worker),
            daemon=True,
        )
        thread.start()
        return dict(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {
                    "job_id": job_id,
                    "description": "",
                    "status": "missing",
                    "stage": "missing",
                    "progress": 0,
                    "message": "未找到对应入库任务。",
                    "result": None,
                    "error": "",
                    "created_at": "",
                    "started_at": "",
                    "finished_at": "",
                }
            return dict(job)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id] = {**self._jobs[job_id], **changes}

    def _run_job(self, job_id: str, worker) -> None:
        self._update(
            job_id,
            status="running",
            stage="starting",
            progress=5,
            message="正在解析 PDF 并写入知识库。",
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

        def progress_callback(stage: str, message: str, progress: int) -> None:
            self._update(
                job_id,
                status="running",
                stage=stage,
                message=message,
                progress=max(0, min(int(progress), 99)),
            )

        try:
            if inspect.signature(worker).parameters:
                result = worker(progress_callback)
            else:
                result = worker()
            success = bool(result.get("success")) if isinstance(result, dict) else True
            message = str(result.get("message", "")) if isinstance(result, dict) else "入库任务已完成。"
            self._update(
                job_id,
                status="succeeded" if success else "failed",
                stage="completed" if success else "failed",
                progress=100,
                message=message or ("入库任务已完成。" if success else "入库任务失败。"),
                result=result,
                error="" if success else message,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                message=f"入库任务失败：{exc}",
                result=None,
                error=str(exc),
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )


class DocumentMetadataExtractionResult(BaseModel):
    """约束文档入库时的标题和摘要抽取结果。"""

    title: str  # 论文标题；抽取失败时上层会退回文件名。
    summary: str  # 论文摘要；用于文档列表和入库概览展示。


# ── 元数据注册（documents.json） ──────────────────────────────────

def _load_docs() -> dict:
    if DOCS_FILE.exists():
        return json.loads(DOCS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_docs(data: dict):
    DOCS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")




_DEFAULT_INGESTION_PROFILE = "default"
_VISUAL_INGESTION_PROFILE = "visual"


def _ingestion_profile(include_visual_descriptions: bool) -> str:
    return _VISUAL_INGESTION_PROFILE if include_visual_descriptions else _DEFAULT_INGESTION_PROFILE


def _is_same_registered_document(filename: str, file_hash: str, ingestion_profile: str) -> bool:
    doc = _load_docs().get(filename)
    if not isinstance(doc, dict):
        return False
    return (
        str(doc.get("file_hash", "")).strip() == file_hash
        and str(doc.get("ingestion_profile", "default")).strip() == ingestion_profile
    )


def _has_registered_document(filename: str) -> bool:
    return isinstance(_load_docs().get(filename), dict)


def _delete_existing_vector_docs(pdf_store, filename: str) -> int:
    """删除同一 source 的旧 Qdrant points，避免同名 PDF 重入库后检索混入旧内容。"""
    client = getattr(pdf_store, "client", None)
    if client is None:
        return 0

    collection_name = str(getattr(pdf_store, "collection_name", PDF_COLLECTION_NAME))
    scroll_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))])
    point_ids = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            point_ids.append(point.id)
        if offset is None:
            break

    if not point_ids:
        return 0

    for start in range(0, len(point_ids), 256):
        batch = point_ids[start:start + 256]
        try:
            client.delete(collection_name=collection_name, points_selector=batch)
        except Exception:
            from qdrant_client.models import PointIdsList
            client.delete(collection_name=collection_name, points_selector=PointIdsList(points=batch))
    return len(point_ids)


def _replace_existing_pdf_records(pdf_store, lexical_index, filename: str) -> int:
    deleted_count = _delete_existing_vector_docs(pdf_store, filename)
    if lexical_index is not None:
        lexical_index.remove_source(PDF_SHARED_SCOPE, filename)
    return deleted_count


def register_document(
    filename: str,
    title: str,
    summary: str,
    chunk_count: int,
    file_hash: str = "",
    ingestion_profile: str = _DEFAULT_INGESTION_PROFILE,
):
    data = _load_docs()
    data[filename] = {
        "filename": filename,
        "title": title,
        "summary": summary,
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chunk_count": chunk_count,
        "file_hash": file_hash,
        "ingestion_profile": ingestion_profile,
    }
    _save_docs(data)
    print(f"[Document] 已注册: {title}")


def is_registered(filename: str) -> bool:
    return filename in _load_docs()


def get_all_documents() -> list:
    data = _load_docs()
    return sorted(data.values(), key=lambda x: x["date_added"], reverse=True)


def format_doc_list() -> str:
    docs = get_all_documents()
    if not docs:
        return "知识库中暂无文档，请先上传 PDF 文件。"
    lines = [f"知识库中共有 {len(docs)} 篇文档：\n"]
    for i, d in enumerate(docs, 1):
        lines.append(
            f"{i}. {d['title']}\n"
            f"   文件名：{d['filename']} | 入库时间：{d['date_added']}\n"
            f"   摘要：{d['summary']}\n"
        )
    return "\n".join(lines)


def ensure_document_registry_from_vector_store(pdf_store) -> None:
    """当 documents.json 缺失或不完整时，从共享 PDF 向量库回填最小文档清单。"""
    global _DOCUMENT_REGISTRY_READY
    if _DOCUMENT_REGISTRY_READY:
        return

    data = _load_docs()
    if data:
        _DOCUMENT_REGISTRY_READY = True
        return
    source_stats: dict[str, int] = {}
    offset = None

    # 这里按 source 聚合现有 chunk，目的是让旧知识库在没有 documents.json 的情况下，
    # 也能恢复出最小可用的文档列表，而不用要求用户重新上传 PDF。
    while True:
        points, offset = pdf_store.client.scroll(
            collection_name="pdf_knowledge",
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = _extract_payload_metadata(payload)
            source_name = str(metadata.get("source", "")).strip()
            if not source_name:
                continue
            source_stats[source_name] = source_stats.get(source_name, 0) + 1
        if offset is None:
            break

    changed = False
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for source_name, chunk_count in source_stats.items():
        if source_name in data:
            continue
        data[source_name] = {
            "filename": source_name,
            "title": source_name,
            "summary": REGISTRY_FALLBACK_SUMMARY,
            "date_added": now,
            "chunk_count": chunk_count,
        }
        changed = True

    if changed:
        _save_docs(data)
        print(f"[Document] 已从共享知识库回填 {sum(1 for key in source_stats if key in data)} 条文档元数据")


# ── PDF 入库 ──────────────────────────────────────────────────────

def _build_child_docs_from_pages(filename: str, pages: list[PageContent]) -> list:
    """按页构建子块，确保每个 chunk 继承稳定的页级证据 metadata。

    边界页的子块会附带 next_parent_id / prev_parent_id，用于检索时跨页补充上下文。
    """
    parent_splitter = _PARENT_SPLITTER
    child_splitter = _CHILD_SPLITTER

    page_parent_ids: dict[int, list[str]] = {}  # page_number → [parent_id 按顺序]
    child_docs = []
    for page in sorted(pages, key=lambda item: item.page_number):
        page_text = page.text.strip()
        if not page_text:
            continue
        parent_ids: list[str] = []
        for p_doc in parent_splitter.create_documents([page_text]):
            p_id = str(uuid.uuid4())
            parent_ids.append(p_id)
            p_text = p_doc.page_content
            for child in child_splitter.create_documents([p_text]):
                chunk_id = build_chunk_id(filename, p_id, child.page_content)
                child.metadata = {
                    "chunk_id": chunk_id,
                    "parent_id": p_id,
                    "parent_text": p_text,
                    **page.metadata(),
                    "source": filename,
                }
                child_docs.append(child)
        if parent_ids:
            page_parent_ids[page.page_number] = parent_ids

    # 给边界页的子块补齐跨页链接
    if len(page_parent_ids) >= 2:
        for child_doc in child_docs:
            pid = child_doc.metadata.get("parent_id")
            pn = int(child_doc.metadata.get("page_number", 0))
            if not pid or pn not in page_parent_ids:
                continue

            page_pids = page_parent_ids[pn]
            # 当前父块是此页最后一个 parent，且下一页存在 → 链接到下一页第一个 parent
            if pid == page_pids[-1]:
                next_pids = page_parent_ids.get(pn + 1)
                if next_pids:
                    child_doc.metadata["next_parent_id"] = next_pids[0]

            # 当前父块是此页第一个 parent，且前一页存在 → 链接到前一页最后一个 parent
            if pid == page_pids[0]:
                prev_pids = page_parent_ids.get(pn - 1)
                if prev_pids:
                    child_doc.metadata["prev_parent_id"] = prev_pids[-1]

    return child_docs

def load_document(
    pdf_path: str,
    pdf_store,
    user_id: str,
    fast_llm=None,
    lexical_index=None,
    recovery_manager: ModelRecoveryManager | None = None,
    include_visual_descriptions: bool = False,
    vision_analyzer=None,
    progress_callback=None,
) -> dict:
    """解析 PDF 并写入共享知识库，返回操作结果字典。"""
    if not os.path.exists(pdf_path):
        return {"success": False, "message": f"文件不存在: {pdf_path}"}

    filename = os.path.basename(pdf_path)
    file_hash = _file_sha256(pdf_path)
    ingestion_profile = _ingestion_profile(include_visual_descriptions)
    if progress_callback:
        progress_callback("detecting", "正在检查 PDF 是否已入库。", 5)
    if _is_same_registered_document(filename, file_hash, ingestion_profile):
        if progress_callback:
            progress_callback("completed", "文档已入库，复用已有处理结果。", 100)
        return {
            "success": True,
            "message": f"文档已入库，复用已有处理结果：{filename}",
            "document": filename,
            "reused": True,
        }

    start = time.time()
    try:
        replace_existing = _has_registered_document(filename)
        if replace_existing:
            if progress_callback:
                progress_callback("cleanup", "正在清理同名旧 PDF 的检索索引。", 15)
            _replace_existing_pdf_records(pdf_store, lexical_index, filename)

        if progress_callback:
            progress_callback("parsing", "正在按页解析 PDF 内容。", 25)
        page_contents = process_pdf_pages(
            pdf_path,
            include_visual_descriptions=include_visual_descriptions,
            vision_analyzer=vision_analyzer,
        )
        md_text = pages_to_markdown(page_contents)
        child_docs = _build_child_docs_from_pages(filename, page_contents)

        if progress_callback:
            progress_callback("indexing", "正在写入向量库和 BM25 索引。", 75)
        pdf_store.add_documents(child_docs)
        if lexical_index is not None:
            lexical_index.index_documents(child_docs, user_id=PDF_SHARED_SCOPE, source=filename)
        elapsed = time.time() - start

        if not _is_same_registered_document(filename, file_hash, ingestion_profile):
            if progress_callback:
                progress_callback("registering", "正在写入文档登记信息。", 90)
            if fast_llm:
                _register_with_llm(
                    filename,
                    md_text,
                    len(child_docs),
                    fast_llm,
                    recovery_manager=recovery_manager,
                    file_hash=file_hash,
                    ingestion_profile=ingestion_profile,
                )
            else:
                register_document(
                    filename,
                    title=filename,
                    summary="（未生成摘要）",
                    chunk_count=len(child_docs),
                    file_hash=file_hash,
                    ingestion_profile=ingestion_profile,
                )

        if progress_callback:
            progress_callback("completed", "PDF 入库完成。", 100)
        return {
            "success":  True,
            "message":  f"解析成功 (耗时: {elapsed:.1f}s)，入库 {len(child_docs)} 个子块",
            "document": filename,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


def _register_with_llm(
    filename: str,
    md_text: str,
    chunk_count: int,
    fast_llm,
    recovery_manager: ModelRecoveryManager | None = None,
    file_hash: str = "",
    ingestion_profile: str = _DEFAULT_INGESTION_PROFILE,
):
    """在文档首次入库时抽取标题和摘要，并写入 documents.json。"""
    excerpt = md_text[:3000]
    prompt = (
        "请从以下学术论文内容中提取文档元数据。\n"
        "要求：\n"
        "1. title 尽量提取论文完整标题；如果内容里无法确定，就返回文件名。\n"
        "2. summary 用 2-3 句话概括论文的研究问题、方法和主要贡献。\n"
        "3. 不要输出额外解释，只返回 schema 对应字段。\n\n"
        f"文件名：{filename}\n"
        f"论文内容：\n{excerpt}"
    )
    recovery_manager = recovery_manager or ModelRecoveryManager()
    extraction_result = recovery_manager.invoke_structured_model(
        fast_llm,
        DocumentMetadataExtractionResult,
        prompt,
        purpose="文档元数据抽取",
    )
    if extraction_result.ok:
        metadata_result = extraction_result.value
        title = metadata_result.title.strip() or filename
        summary = metadata_result.summary.strip() or "（未能提取摘要）"
    else:
        title, summary = filename, "（未能提取摘要）"
    register_document(
        filename,
        title=title,
        summary=summary,
        chunk_count=chunk_count,
        file_hash=file_hash,
        ingestion_profile=ingestion_profile,
    )
