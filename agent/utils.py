"""跨模块共享的工具函数。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """计算文件 SHA-256 哈希值，分块读取避免大文件内存峰值。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_payload_metadata(payload: dict | None) -> dict:
    """从 Qdrant payload 中提取 metadata 字典。

    兼容两种存储格式：payload 直接作为 metadata（扁平格式）或 payload.metadata 嵌套。
    """
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else payload


def clean_query(query: str) -> str:
    """规范化查询字符串：去除首尾空白并合并连续空格。"""
    return " ".join(str(query).strip().split())
