# lexical_index.py
# 持久化 lexical index，专门服务论文正文的 BM25 检索。
# 这层和向量库分开维护：
# - 向量库负责语义召回
# - lexical index 负责术语、缩写、模块名、数据集名等精确匹配

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

from agent.memory import PDF_COLLECTION_NAME
from agent.utils import extract_payload_metadata
from langchain_core.documents import Document

INDEX_DB_PATH = Path(__file__).parent / "lexical_index.db"  # 持久化 BM25 倒排索引文件。
BM25_K1 = 1.5  # BM25 术语频率增益参数。
BM25_B = 0.75  # BM25 文档长度归一化参数。
PDF_SHARED_SCOPE = "__shared_pdf__"  # 共享 PDF 知识库在 lexical index 中使用的固定作用域标识。


def _tokenize_for_bm25(text: str) -> list[str]:
    """为 BM25 生成一套兼顾英文术语和中文检索的轻量 token。"""
    normalized = text.lower().strip()
    if not normalized:
        return []

    ascii_tokens = re.findall(r"[a-z0-9][a-z0-9_\-./]*", normalized)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_bigrams = [cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)]
    return ascii_tokens + cjk_chars + cjk_bigrams


def build_chunk_id(source: str, parent_id: str, content: str) -> str:
    """用稳定 hash 生成共享 PDF child chunk id，保证向量库和 lexical index 可以对齐。"""
    raw_key = f"{source}\n{parent_id}\n{content}".encode("utf-8")
    return hashlib.sha1(raw_key).hexdigest()


class PersistentLexicalIndex:
    """基于 SQLite 的轻量持久化 BM25 索引。"""

    db_path: Path  # lexical index 的 SQLite 文件路径。

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or INDEX_DB_PATH
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """创建数据库连接，并打开外键约束。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        """初始化 chunk 表和 postings 倒排表。"""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    parent_text TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    doc_len INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS postings (
                    token TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    term_freq INTEGER NOT NULL,
                    PRIMARY KEY (token, chunk_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_user_source ON chunks(user_id, source);
                CREATE INDEX IF NOT EXISTS idx_postings_token ON postings(token);
                CREATE INDEX IF NOT EXISTS idx_postings_chunk_id ON postings(chunk_id);
                """
            )
            chunk_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "metadata_json" not in chunk_columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")

    def remove_source(self, user_id: str, source: str) -> None:
        """删除某个 source 下的全部共享 PDF lexical index 数据。"""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE source = ?",
                (source,),
            )

    def index_documents(
        self,
        child_docs: list[Document],
        user_id: str,
        source: str,
        replace_source: bool = True,
    ) -> None:
        """把新入库的 child chunks 写入持久化 lexical index。"""
        if not child_docs:
            return

        if replace_source:
            self.remove_source(user_id, source)

        chunk_rows: list[tuple[str, str, str, str, str, str, str, int]] = []
        posting_rows: list[tuple[str, str, int]] = []

        # 这里在入库阶段就把 token postings 写好，避免检索时再全量扫正文计算 BM25。
        for doc in child_docs:
            parent_id = str(doc.metadata.get("parent_id", ""))
            parent_text = str(doc.metadata.get("parent_text", doc.page_content))
            content = str(doc.page_content)
            chunk_id = str(doc.metadata.get("chunk_id") or build_chunk_id(source, parent_id, content))
            tokens = _tokenize_for_bm25(content)
            term_frequency = Counter(tokens)

            chunk_rows.append(
                (
                    chunk_id,
                    PDF_SHARED_SCOPE,
                    source,
                    parent_id,
                    parent_text,
                    content,
                    json.dumps(doc.metadata, ensure_ascii=False),
                    len(tokens),
                )
            )
            posting_rows.extend((token, chunk_id, frequency) for token, frequency in term_frequency.items())

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, user_id, source, parent_id, parent_text, content, metadata_json, doc_len
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )
            conn.executemany(
                "INSERT OR REPLACE INTO postings (token, chunk_id, term_freq) VALUES (?, ?, ?)",
                posting_rows,
            )

    def ensure_source_indexed_from_vector_store(self, pdf_store, user_id: str, source: str = "") -> None:
        """兼容旧数据：当 lexical index 缺失时，从共享 PDF 向量库扫描并补建。"""
        existing_sources = self._get_indexed_sources(user_id, source)
        if source and source in existing_sources:
            return

        child_docs_by_source: dict[str, list[Document]] = {}
        offset = None

        while True:
            points, offset = pdf_store.client.scroll(
                collection_name=PDF_COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                metadata = extract_payload_metadata(point.payload or {})
                point_source = str(metadata.get("source", ""))
                if source and point_source != source:
                    continue
                if point_source in existing_sources:
                    continue

                page_content = str(payload.get("page_content", "")).strip()
                if not page_content:
                    continue
                metadata["chunk_id"] = build_chunk_id(
                    point_source,
                    str(metadata.get("parent_id", "")),
                    page_content,
                )
                child_docs_by_source.setdefault(point_source, []).append(
                    Document(page_content=page_content, metadata=metadata)
                )

            if offset is None:
                break

        for point_source, child_docs in child_docs_by_source.items():
            self.index_documents(child_docs, user_id=PDF_SHARED_SCOPE, source=point_source, replace_source=False)

    def _get_indexed_sources(self, user_id: str, source: str = "") -> set[str]:
        """查询共享 PDF 知识库里哪些 source 已经建立 lexical index。"""
        sql = "SELECT DISTINCT source FROM chunks WHERE user_id = ?"
        args: list[str] = [PDF_SHARED_SCOPE]
        if source:
            sql += " AND source = ?"
            args.append(source)

        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return {str(row["source"]) for row in rows}

    def search(self, query: str, user_id: str, source: str = "", top_k: int = 3) -> list[Document]:
        """在共享 PDF 的持久化 lexical index 上执行 BM25 检索。"""
        query_tokens = list(dict.fromkeys(_tokenize_for_bm25(query)))
        if not query_tokens:
            return []

        sql_filter = "c.user_id = ?"
        args: list[str] = [PDF_SHARED_SCOPE]
        if source:
            sql_filter += " AND c.source = ?"
            args.append(source)

        with self._connect() as conn:
            corpus_stats = conn.execute(
                f"SELECT COUNT(*) AS doc_count, AVG(doc_len) AS avg_doc_len FROM chunks c WHERE {sql_filter}",
                args,
            ).fetchone()
            total_docs = int(corpus_stats["doc_count"] or 0)
            avg_doc_len = float(corpus_stats["avg_doc_len"] or 0.0)
            if total_docs == 0:
                return []

            placeholders = ",".join("?" for _ in query_tokens)
            posting_rows = conn.execute(
                f"""
                SELECT
                    p.token,
                    p.chunk_id,
                    p.term_freq,
                    c.doc_len,
                    c.content,
                    c.parent_id,
                    c.parent_text,
                    c.source,
                    c.metadata_json
                FROM postings p
                JOIN chunks c ON c.chunk_id = p.chunk_id
                WHERE {sql_filter} AND p.token IN ({placeholders})
                """,
                args + query_tokens,
            ).fetchall()

        if not posting_rows:
            return []

        doc_frequency = Counter()
        chunk_rows_by_id: dict[str, sqlite3.Row] = {}
        for row in posting_rows:
            doc_frequency[str(row["token"])] += 1
            chunk_rows_by_id[str(row["chunk_id"])] = row

        scores: dict[str, float] = {}
        for row in posting_rows:
            token = str(row["token"])
            chunk_id = str(row["chunk_id"])
            term_freq = int(row["term_freq"])
            doc_len = int(row["doc_len"] or 0)
            idf = math.log(1 + (total_docs - doc_frequency[token] + 0.5) / (doc_frequency[token] + 0.5))
            numerator = term_freq * (BM25_K1 + 1)
            denominator = term_freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(avg_doc_len, 1.0))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * numerator / max(denominator, 1e-6)

        ranked_chunk_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        ranked_docs: list[Document] = []
        for chunk_id in ranked_chunk_ids:
            row = chunk_rows_by_id[chunk_id]
            try:
                stored_metadata = json.loads(str(row["metadata_json"] or "{}"))
                if not isinstance(stored_metadata, dict):
                    stored_metadata = {}
            except json.JSONDecodeError:
                stored_metadata = {}
            stored_metadata.update(
                {
                    "chunk_id": chunk_id,
                    "parent_id": str(row["parent_id"]),
                    "parent_text": str(row["parent_text"]),
                    "source": str(row["source"]),
                    "user_id": PDF_SHARED_SCOPE,
                }
            )
            ranked_docs.append(
                Document(
                    page_content=str(row["content"]),
                    metadata=stored_metadata,
                )
            )

        return ranked_docs
