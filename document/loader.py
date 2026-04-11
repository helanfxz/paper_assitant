# document/loader.py
# 负责 PDF 文档的解析和向量入库。
# 使用父子块策略：父块（1500字）保留完整上下文，子块（400字）用于精细检索。
# 入库完成后调用 fast_llm 提取论文标题和摘要，注册到 documents.json 持久化。

import os
import uuid
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm
from document.registry import register_document, is_registered


def load_document(pdf_path: str, pdf_store, user_id: str, fast_llm=None) -> dict:
    """解析 PDF 并以父子块策略入库，返回操作结果字典。"""
    if not os.path.exists(pdf_path):
        return {"success": False, "message": f"文件不存在: {pdf_path}"}

    filename = os.path.basename(pdf_path)
    start_time = time.time()
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        parent_docs = parent_splitter.create_documents([md_text])
        child_docs = []
        for p_doc in parent_docs:
            p_id = str(uuid.uuid4())
            p_text = p_doc.page_content
            for c in child_splitter.create_documents([p_text]):
                c.metadata = {
                    "parent_id": p_id,
                    "parent_text": p_text,
                    "source": filename,
                    "user_id": user_id,
                }
                child_docs.append(c)

        pdf_store.add_documents(child_docs)
        elapsed = time.time() - start_time

        # 提取标题和摘要，注册到 documents.json
        if fast_llm and not is_registered(filename):
            _register_metadata(filename, md_text, len(child_docs), fast_llm)
        elif not fast_llm and not is_registered(filename):
            register_document(filename, title=filename, summary="（未生成摘要）", chunk_count=len(child_docs))

        return {
            "success": True,
            "message": f"解析成功 (耗时: {elapsed:.1f}s)，入库 {len(child_docs)} 个子块",
            "document": filename,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


def _register_metadata(filename: str, md_text: str, chunk_count: int, fast_llm):
    """用 fast_llm 从文档开头提取标题和生成摘要。"""
    # 取前 3000 字，通常包含标题、摘要、引言
    excerpt = md_text[:3000]
    prompt = (
        f"请从以下学术论文内容中提取信息，严格按格式输出：\n"
        f"标题：<论文完整标题，如无则用文件名>\n"
        f"摘要：<2-3句话概括论文的研究问题、方法和主要贡献>\n\n"
        f"论文内容：\n{excerpt}"
    )
    try:
        raw = fast_llm.invoke(prompt).content
        title_line = next((l for l in raw.splitlines() if "标题" in l), "")
        summary_line = next((l for l in raw.splitlines() if "摘要" in l), "")
        title = title_line.split("：", 1)[-1].strip() if title_line else filename
        summary = summary_line.split("：", 1)[-1].strip() if summary_line else "（未能提取摘要）"
    except Exception:
        title, summary = filename, "（未能提取摘要）"

    register_document(filename, title=title, summary=summary, chunk_count=chunk_count)
