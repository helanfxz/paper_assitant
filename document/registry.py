# document/registry.py
# 负责文档元数据的持久化管理。
# 每次 PDF 入库时，提取标题、生成摘要、记录入库时间，存入 documents.json。
# list_documents 工具从此处读取，确保重启后数据不丢失。

import json
import os
from datetime import datetime

DOCS_FILE = os.path.join(os.path.dirname(__file__), "..", "documents.json")


def _load() -> dict:
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data: dict):
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def register_document(filename: str, title: str, summary: str, chunk_count: int):
    """将文档元数据写入 documents.json。"""
    data = _load()
    data[filename] = {
        "filename": filename,
        "title": title,
        "summary": summary,
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chunk_count": chunk_count,
    }
    _save(data)
    print(f"[Registry] 文档已注册: {title}")


def get_all_documents() -> list:
    """返回所有已注册文档的元数据列表，按入库时间倒序。"""
    data = _load()
    return sorted(data.values(), key=lambda x: x["date_added"], reverse=True)


def is_registered(filename: str) -> bool:
    return filename in _load()


def format_doc_list() -> str:
    """格式化为 list_documents 工具的返回字符串。"""
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
