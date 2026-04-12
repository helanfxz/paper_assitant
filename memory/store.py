# memory/store.py
# 负责 Qdrant 向量数据库的初始化，以及长期记忆的读写操作。
# 提供 build_memory_context() 在每轮对话前检索相关记忆注入上下文，
# 以及 save_fact() 将重要事实持久化到 Qdrant。

import os
from datetime import datetime
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def init_qdrant(embeddings) -> tuple[QdrantClient, QdrantVectorStore, QdrantVectorStore]:
    """初始化 Qdrant 客户端和两个向量库，返回 (client, pdf_store, memory_store)。"""
    qdrant_url = os.getenv("QDRANT_URL", ":memory:")
    client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))

    existing = [col.name for col in client.get_collections().collections]
    for col_name in ["pdf_knowledge", "user_semantic_memory"]:
        if col_name not in existing:
            client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
            )

    pdf_store = QdrantVectorStore(
        client=client, collection_name="pdf_knowledge", embedding=embeddings
    )
    memory_store = QdrantVectorStore(
        client=client, collection_name="user_semantic_memory", embedding=embeddings
    )
    return client, pdf_store, memory_store


def build_memory_context(memory_store: QdrantVectorStore, user_id: str, session_id: str, question: str) -> str:
    """用当前问题语义检索该 session 的长期记忆，综合相似度和近期性排序后注入上下文。"""
    try:
        results = memory_store.similarity_search_with_score(
            question, k=10,
            filter={"must": [
                {"key": "user_id", "match": {"value": user_id}},
                {"key": "session_id", "match": {"value": session_id}},
            ]}
        )
    except Exception:
        results = []

    if not results:
        return ""

    now = datetime.now()
    scored = []
    for doc, similarity in results:
        ts = doc.metadata.get("timestamp", "")
        try:
            days_ago = (now - datetime.fromisoformat(ts)).days
        except Exception:
            days_ago = 30
        recency = 1 / (1 + days_ago)
        score = 0.7 * similarity + 0.3 * recency
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored[:3] if score >= 0.5]

    if not top_docs:
        return ""

    facts = [d for d in top_docs if d.metadata.get("type") == "auto_fact"]
    notes = [d for d in top_docs if d.metadata.get("type") == "note"]

    parts = ["【本会话的长期记忆】："]
    if facts:
        parts.append("[事实]")
        parts.extend(f"- {d.page_content}" for d in facts)
    if notes:
        parts.append("[笔记]")
        parts.extend(f"- {d.page_content}" for d in notes)
    parts.append("请在回答时参考以上背景。")

    return "\n".join(parts)


def save_fact(memory_store: QdrantVectorStore, content: str, user_id: str,
              session_id: str, fact_type: str = "auto_fact"):
    """将一条事实存入 Qdrant，按 user_id + session_id 隔离。"""
    doc = Document(
        page_content=content,
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "type": fact_type,
            "timestamp": datetime.now().isoformat(),
        }
    )
    memory_store.add_documents([doc])
