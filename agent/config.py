"""统一管理模型、Embedding 和 reranker 初始化。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from agent.reranker import DEFAULT_DASHSCOPE_RERANK_URL, DashScopeReranker

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 项目根目录。

load_dotenv(PROJECT_ROOT / ".env")

# ── 扫描型 PDF 处理配置 ──
SCANNED_PAGE_CHAR_THRESHOLD = int(os.getenv("SCANNED_PAGE_CHAR_THRESHOLD", "50"))
PDF_TO_IMAGE_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_LANG = os.getenv("OCR_LANG", "ch")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() == "true"
OCR_FALLBACK_LANG = os.getenv("OCR_FALLBACK_LANG", "chi_sim+eng")
GLM4V_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "1024"))
GLM4V_TIMEOUT_SECONDS = int(os.getenv("VISION_TIMEOUT", "30"))
PDF_PAGE_CACHE_ENABLED = os.getenv("PDF_PAGE_CACHE_ENABLED", "true").lower() not in {"0", "false", "no"}
PDF_PAGE_CACHE_DIR = Path(os.getenv("PDF_PAGE_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "pdf_pages")))

LLM_TIMEOUT_SECONDS = 60  # 主模型请求超时时间。
FAST_LLM_TIMEOUT_SECONDS = 30  # 轻量模型请求超时时间。
RERANK_TIMEOUT_SECONDS = 20  # rerank HTTP 请求超时时间。
_LLM = None  # 主模型实例单例缓存。
_FAST_LLM = None  # fast_llm 实例单例缓存。
_RERANKER = None  # reranker 实例单例缓存。
_EMBEDDINGS = None  # embedding 实例单例缓存。
_VISION_LLM = None  # 视觉模型实例单例缓存。


def get_llm():
    """创建主模型实例。"""
    global _LLM
    if _LLM is not None:
        return _LLM
    _LLM = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )
    return _LLM


def get_fast_llm():
    """创建轻量模型实例。"""
    global _FAST_LLM
    if _FAST_LLM is not None:
        return _FAST_LLM
    _FAST_LLM = ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        temperature=1.0,
        max_tokens=65536,
        timeout=FAST_LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )
    return _FAST_LLM


def _get_rerank_url() -> str:
    """读取 rerank 地址，并把 DashScope 兼容模式地址纠正为 text-rerank 专用地址。"""
    configured_url = os.getenv("RERANK_URL", "").strip()
    if not configured_url:
        return DEFAULT_DASHSCOPE_RERANK_URL

    # DashScope compatible-api/v1/reranks 不是当前 wrapper 使用的 text-rerank 协议。
    # 用户如果配置了这个地址，按 qwen3-rerank 官方示例自动切回专用 endpoint。
    if "compatible-api" in configured_url and configured_url.rstrip("/").endswith("/reranks"):
        print(
            "[Rerank] 当前 RERANK_URL 是 DashScope compatible-api reranks，"
            "已自动使用 text-rerank 专用 endpoint。"
        )
        return DEFAULT_DASHSCOPE_RERANK_URL

    return configured_url


def get_reranker():
    """创建 DashScope rerank 客户端。

    rerank 模型使用 query/documents 协议，不是 chat/completions 协议，
    所以这里返回专用客户端，而不是 ChatOpenAI。
    """
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER

    api_key = os.getenv("RERANK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 RERANK_API_KEY 或 DASHSCOPE_API_KEY，无法初始化 reranker。")

    _RERANKER = DashScopeReranker(
        model=os.getenv("RERANK_MODEL_NAME", "qwen3-rerank"),
        api_key=api_key,
        url=_get_rerank_url(),
        timeout=RERANK_TIMEOUT_SECONDS,
    )
    return _RERANKER


def get_rerank_llm():
    """兼容旧引用：当前返回的是专用 reranker，不再是 ChatOpenAI。"""
    return get_reranker()


def get_embeddings():
    """创建 Embedding 模型实例。"""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS
    _EMBEDDINGS = OpenAIEmbeddings(
        model="Embedding-3",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        check_embedding_ctx_length=False,
    )
    return _EMBEDDINGS


def get_vision_llm():
    """创建 GLM-4V-Flash 多模态模型实例，用于图表描述。复用智谱 API Key。"""
    global _VISION_LLM
    if _VISION_LLM is not None:
        return _VISION_LLM
    _VISION_LLM = ChatOpenAI(
        model=os.getenv("VISION_MODEL", "glm-4v-flash"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        max_tokens=GLM4V_MAX_TOKENS,
        timeout=GLM4V_TIMEOUT_SECONDS,
        max_retries=0,
    )
    return _VISION_LLM
