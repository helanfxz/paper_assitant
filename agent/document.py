# document.py
# PDF 文档的解析入库 + 元数据注册。
# 父子块策略：父块（1500字）保留完整上下文，子块（400字）用于精细检索。
# documents.json 持久化文档元数据，重启后不丢失。

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from agent.error_recovery import ModelRecoveryManager
from lexical_index import build_chunk_id

_BASE = Path(__file__).resolve().parent.parent  # 项目根目录，保证 documents.json 仍然落在原位置。
DOCS_FILE = _BASE / "documents.json"
REGISTRY_FALLBACK_SUMMARY = "（从现有知识库回填，未生成摘要）"  # 旧知识库回填 documents.json 时使用的默认摘要。
_DOCUMENT_REGISTRY_READY = False  # 每个进程只做一次文档注册回填，避免反复全库扫描。


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


def register_document(filename: str, title: str, summary: str, chunk_count: int):
    data = _load_docs()
    data[filename] = {
        "filename": filename,
        "title": title,
        "summary": summary,
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chunk_count": chunk_count,
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
            metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else payload
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

def load_document(
    pdf_path: str,
    pdf_store,
    user_id: str,
    fast_llm=None,
    lexical_index=None,
    recovery_manager: ModelRecoveryManager | None = None,
) -> dict:
    """解析 PDF 并写入共享知识库，返回操作结果字典。"""
    if not os.path.exists(pdf_path):
        return {"success": False, "message": f"文件不存在: {pdf_path}"}

    filename = os.path.basename(pdf_path)
    start = time.time()
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=50)

        child_docs = []
        for p_doc in parent_splitter.create_documents([md_text]):
            p_id   = str(uuid.uuid4())
            p_text = p_doc.page_content
            for c in child_splitter.create_documents([p_text]):
                chunk_id = build_chunk_id(filename, p_id, c.page_content)
                c.metadata = {
                    "chunk_id":    chunk_id,
                    "parent_id":   p_id,
                    "parent_text": p_text,
                    "source":      filename,
                }
                child_docs.append(c)

        pdf_store.add_documents(child_docs)
        if lexical_index is not None:
            lexical_index.index_documents(child_docs, user_id="__shared_pdf__", source=filename)
        elapsed = time.time() - start

        if not is_registered(filename):
            if fast_llm:
                _register_with_llm(
                    filename,
                    md_text,
                    len(child_docs),
                    fast_llm,
                    recovery_manager=recovery_manager,
                )
            else:
                register_document(filename, title=filename, summary="（未生成摘要）",
                                  chunk_count=len(child_docs))

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
    register_document(filename, title=title, summary=summary, chunk_count=chunk_count)
