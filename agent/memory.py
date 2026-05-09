# memory.py
# 记忆层，负责三类能力：
# 1. Qdrant 向量存储初始化
# 2. 当前 session 的语义记忆检索与窗口压缩
# 3. 跨 session 的用户偏好状态（Profile）
#
# 这里的 Profile 不做“可检索碎片记忆”，而是做“当前生效的用户偏好状态”：
# - 只保存少量跨 session 的稳定偏好
# - 每个偏好通过 scope + type 定位到固定槽位
# - 同槽位新值覆盖旧值，避免 prompt 中出现冲突偏好

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
from agent.error_recovery import ModelRecoveryManager

# Qdrant collection 名称。
PDF_COLLECTION_NAME = "pdf_knowledge"  # 论文分块向量库集合名。
SESSION_MEMORY_COLLECTION_NAME = "user_semantic_memory"  # 当前 session 的窗口压缩摘要集合名。
STUDY_NOTES_COLLECTION_NAME = "study_notes"  # 跨 session 的用户研究笔记集合名。

# Session memory 召回与压缩参数。
# 这些参数决定了：
# 1. 长对话何时开始把旧消息压出窗口
# 2. 一次压缩多少条可见对话
# 3. 召回时最低要达到多少相关度才值得重新注入 prompt
SESSION_MEMORY_SCORE_THRESHOLD = 0.5  # 召回结果最低分数阈值，低于它的摘要或笔记直接丢弃。
SESSION_WINDOW_SIZE = 20  # 对话窗口超过这个条数后，开始把更早的内容压缩成 session summary。
SESSION_COMPRESS_BATCH_SIZE = 4  # 每次从窗口最前面拿多少条可见对话做一次压缩。
SESSION_SUMMARY_TYPE = "session_summary"  # 新版窗口压缩摘要的类型标签。
LEGACY_AUTO_FACT_TYPE = "auto_fact"  # 兼容旧数据时识别的历史摘要类型标签。
STUDY_NOTES_SCORE_THRESHOLD = 0.55  # 研究笔记自动召回时的最低分数阈值。
STUDY_NOTES_RECALL_LIMIT = 3  # 每轮自动注入 prompt 的研究笔记上限。
_VECTOR_STORES_CACHE = None  # 向量库客户端与 store 单例缓存，避免每次建会话都重复初始化。

# Profile 存储与枚举常量。
# 这里定义的是 profile 这套“长期偏好状态机”允许出现的固定槽位和值域，
# 后续偏好检测、状态更新、prompt 注入都会复用这些枚举。
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 项目根目录，供 agent/ 下模块回到统一数据根路径。
USER_PROFILE_ROOT = PROJECT_ROOT / "data" / "profiles"  # 每个用户的 profile 状态和事件日志根目录。
PROFILE_SCOPES = ("global", "paper_summary", "paper_compare")  # 支持的偏好作用范围。
PROFILE_TYPES = ("language", "format", "detail_level", "focus", "avoid")  # 支持的偏好槽位类型。
PROFILE_OPERATIONS = ("add", "update", "delete", "clear")  # 允许的 profile 管理操作。
PROFILE_CONFIDENCE = ("high", "medium", "low")  # 偏好检测结果的置信度档位。

SCOPE_LABELS = {  # 偏好 scope 在日志和 prompt 中使用的人类可读标签。
    "global": "全局回答",
    "paper_summary": "论文总结",
    "paper_compare": "论文对比",
}

TYPE_LABELS = {  # 偏好 type 在日志和 prompt 中使用的人类可读标签。
    "language": "语言偏好",
    "format": "输出格式",
    "detail_level": "详细程度",
    "focus": "重点关注",
    "avoid": "尽量避免",
}

# 这些词只用于“是否值得触发偏好检测”的前置筛选，
# 目的是避免每轮普通问答都额外调用 fast_llm。
LONG_TERM_PREFERENCE_HINTS = (  # 命中这些词时，才值得进一步触发长期偏好检测。
    "以后", "之后", "默认", "每次", "一直", "通常",
    "我更喜欢", "我希望", "请记住", "不要再", "后面都按这个来",
)

# 这些词用于识别“管理 profile”的输入，例如删除、清空、关闭记忆。
PROFILE_MANAGEMENT_HINTS = (  # 命中这些词时，说明用户可能在做 profile 的删改清空操作。
    "删除偏好", "删掉偏好", "清空偏好", "移除偏好",
    "别再记", "不要记录", "关闭偏好记忆",
)
EXPLICIT_PREFERENCE_HINTS = (  # 命中这些短语时，说明用户可能在显式声明或保存长期偏好。
    "记为偏好", "保存偏好", "作为偏好", "长期偏好", "修改偏好", "更改偏好",
)

CONFIRM_TRUE_WORDS = {"是", "好的", "好", "要", "需要", "确认", "可以", "记住吧", "保存吧", "对"}
CONFIRM_FALSE_WORDS = {"不是", "不用", "不要", "否", "算了", "取消", "不用记", "不需要"}

# LLM 输出和用户自然语言里可能会出现同义表达，
# 这些 alias 用来把不同表达归一到系统内部的固定枚举。
SCOPE_ALIASES = {  # 把模型输出或自然语言里的 scope 同义表达归一到内部枚举。
    "global": "global",
    "全局": "global",
    "回答": "global",
    "paper_summary": "paper_summary",
    "summary": "paper_summary",
    "论文总结": "paper_summary",
    "总结论文": "paper_summary",
    "paper_compare": "paper_compare",
    "compare": "paper_compare",
    "论文对比": "paper_compare",
    "对比论文": "paper_compare",
}

TYPE_ALIASES = {  # 把模型输出或自然语言里的 type 同义表达归一到内部枚举。
    "language": "language",
    "语言": "language",
    "format": "format",
    "格式": "format",
    "输出格式": "format",
    "detail": "detail_level",
    "detail_level": "detail_level",
    "详细程度": "detail_level",
    "focus": "focus",
    "关注点": "focus",
    "重点": "focus",
    "avoid": "avoid",
    "忽略项": "avoid",
    "避免": "avoid",
}

OPERATION_ALIASES = {  # 把模型输出或自然语言里的 operation 同义表达归一到内部枚举。
    "add": "add",
    "create": "add",
    "新增": "add",
    "保存": "add",
    "update": "update",
    "modify": "update",
    "修改": "update",
    "覆盖": "update",
    "delete": "delete",
    "remove": "delete",
    "删除": "delete",
    "移除": "delete",
    "clear": "clear",
    "清空": "clear",
}

CONFIDENCE_ALIASES = {  # 把模型输出或自然语言里的 confidence 同义表达归一到内部枚举。
    "high": "high",
    "medium": "medium",
    "low": "low",
    "高": "high",
    "中": "medium",
    "低": "low",
}


class PreferenceDetectionResult(BaseModel):
    """偏好检测结果的结构化模型，只有通过验证的数据才允许写入 Profile。"""

    is_preference: bool  # 这次检测是否认为用户在表达跨 session 的长期偏好。
    scope: Literal["global", "paper_summary", "paper_compare"] | None = None  # 偏好影响的任务范围。
    type: Literal["language", "format", "detail_level", "focus", "avoid"] | None = None  # 偏好写入的槽位类型。
    value: str = ""  # 归一化后的偏好值，真正写入 profile_state 的就是它。
    operation: Literal["add", "update", "delete", "clear"] | None = None  # 这次输入想对 profile 执行的操作。
    confidence: Literal["high", "medium", "low"]  # 偏好检测的置信度档位。
    reason: str = ""  # 记录检测理由，主要用于日志和调试。
    sensitive: bool = False  # 是否属于删除、清空、关闭记忆等敏感操作。


def init_vector_stores(embeddings):
    """初始化 Qdrant 客户端和三个向量库，返回 (client, pdf_store, session_memory_store, study_notes_store)。"""
    global _VECTOR_STORES_CACHE
    if _VECTOR_STORES_CACHE is not None:
        return _VECTOR_STORES_CACHE

    qdrant_url = os.getenv("QDRANT_URL", ":memory:")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    local_qdrant_dir = PROJECT_ROOT / "memory_data" / "qdrant"

    # 鍏堝皾璇曡繙绋?Qdrant銆傚鏋滆繙绋嬪湴鍧€鏃犳晥鎴栨湇鍔′笉鍙敤锛?
    # 灏辫嚜鍔ㄥ洖閫€鍒版湰鍦版寔涔呭寲 Qdrant锛岄伩鍏嶆柊寤轰細璇濇椂鐩存帴澶辫触銆?
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        existing_collections = [collection.name for collection in client.get_collections().collections]
    except Exception as exc:
        local_qdrant_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Qdrant] 杩滅▼鍚戦噺搴撲笉鍙敤锛屽洖閫€鍒版湰鍦? {local_qdrant_dir} | 鍘熷洜: {exc}")
        client = QdrantClient(path=str(local_qdrant_dir))
        existing_collections = [collection.name for collection in client.get_collections().collections]

    # 项目当前维护三个集合：
    # 1. pdf_knowledge：论文原文切块
    # 2. user_semantic_memory：当前 session 的窗口压缩摘要
    # 3. study_notes：跨 session 的研究笔记
    for collection_name in (
        PDF_COLLECTION_NAME,
        SESSION_MEMORY_COLLECTION_NAME,
        STUDY_NOTES_COLLECTION_NAME,
    ):
        if collection_name not in existing_collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
            )

    # 本地 Qdrant 回退目录不会自动带上云端已有的 payload index。
    # 这里统一补齐当前项目检索和过滤实际会用到的 keyword 字段，
    # 避免后续 similarity_search(filter=...) 因缺少索引直接报 400。
    index_fields_by_collection = {
        PDF_COLLECTION_NAME: ("user_id", "source", "parent_id", "chunk_id"),
        SESSION_MEMORY_COLLECTION_NAME: ("user_id", "session_id", "type"),
        STUDY_NOTES_COLLECTION_NAME: ("user_id", "note_id", "source_session_id"),
    }
    for collection_name, field_names in index_fields_by_collection.items():
        for field_name in field_names:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # 索引已存在时 Qdrant 会直接报错，这里按幂等初始化处理。
                pass

    pdf_store = QdrantVectorStore(
        client=client,
        collection_name=PDF_COLLECTION_NAME,
        embedding=embeddings,
    )
    session_memory_store = QdrantVectorStore(
        client=client,
        collection_name=SESSION_MEMORY_COLLECTION_NAME,
        embedding=embeddings,
    )
    study_notes_store = QdrantVectorStore(
        client=client,
        collection_name=STUDY_NOTES_COLLECTION_NAME,
        embedding=embeddings,
    )
    _VECTOR_STORES_CACHE = (client, pdf_store, session_memory_store, study_notes_store)
    return _VECTOR_STORES_CACHE


def _extract_payload_metadata(payload: dict | None) -> dict:
    """兼容 Qdrant payload 的不同形态，统一取出 metadata。"""
    if not isinstance(payload, dict):
        return {}
    nested_metadata = payload.get("metadata")
    return nested_metadata if isinstance(nested_metadata, dict) else payload


def _iter_user_note_records(study_notes_store, user_id: str) -> list[dict[str, str]]:
    """扫描 study_notes 集合，返回当前用户的全部笔记记录。"""
    note_records: list[dict[str, str]] = []
    offset = None

    while True:
        points, offset = study_notes_store.client.scroll(
            collection_name=STUDY_NOTES_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            metadata = _extract_payload_metadata(point.payload)
            if metadata.get("user_id") != user_id:
                continue
            note_records.append(
                {
                    "note_id": str(metadata.get("note_id", point.id)),
                    "content": str(metadata.get("content", "")),
                    "title": str(metadata.get("title", "")),
                    "created_at": str(metadata.get("created_at", "")),
                    "updated_at": str(metadata.get("updated_at", "")),
                    "source_session_id": str(metadata.get("source_session_id", "")),
                }
            )
        if offset is None:
            break

    note_records.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return note_records


def migrate_legacy_notes(session_memory_store, study_notes_store, user_id: str) -> int:
    """把旧 session memory 集合里的 note 条目迁到独立 study_notes 集合。"""
    migrated_count = 0
    existing_note_ids = {note["note_id"] for note in _iter_user_note_records(study_notes_store, user_id)}
    offset = None

    while True:
        points, offset = session_memory_store.client.scroll(
            collection_name=SESSION_MEMORY_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            metadata = _extract_payload_metadata(point.payload)
            if metadata.get("user_id") != user_id or metadata.get("type") != "note":
                continue

            note_id = str(metadata.get("note_id", point.id))
            if note_id in existing_note_ids:
                continue

            raw_payload = point.payload if isinstance(point.payload, dict) else {}
            note_content = str(metadata.get("content", "")).strip()
            if not note_content:
                note_content = str(raw_payload.get("page_content", "")).strip()
            note_title = str(metadata.get("title", "")).strip()
            created_at = str(metadata.get("timestamp", ""))

            note_doc = Document(
                page_content=note_content,
                metadata={
                    "note_id": note_id,
                    "user_id": user_id,
                    "content": note_content,
                    "title": note_title or note_content[:30],
                    "created_at": created_at,
                    "updated_at": created_at,
                    "source_session_id": str(metadata.get("session_id", "")),
                },
            )
            study_notes_store.add_documents([note_doc], ids=[note_id])
            existing_note_ids.add(note_id)
            migrated_count += 1

        if offset is None:
            break

    return migrated_count


def retrieve_session_memory_entries(
    session_memory_store,
    user_id: str,
    session_id: str,
    user_message: str,
) -> list[dict[str, str]]:
    """检索当前 session 内相关的窗口压缩摘要，并按相似度与时间新鲜度重新排序。"""
    try:
        # 这里只在当前用户、当前 session 范围内召回，
        # 避免把别的会话里的摘要或笔记误注入进来。
        recall_results = session_memory_store.similarity_search_with_score(
            user_message,
            k=10,
            filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "session_id", "match": {"value": session_id}},
                ]
            },
        )
    except Exception:
        return []

    now = datetime.now()
    rescored_results = []
    for recalled_doc, similarity_score in recall_results:
        timestamp = recalled_doc.metadata.get("timestamp", "")
        try:
            days_ago = (now - datetime.fromisoformat(timestamp)).days
        except Exception:
            days_ago = 30

        # 当前实现不是只看向量分数，
        # 还会给更新近的 session memory 一点额外权重。
        freshness_bonus = 0.3 / (1 + days_ago)
        final_score = 0.7 * similarity_score + freshness_bonus
        rescored_results.append((recalled_doc, final_score))

    rescored_results.sort(key=lambda item: item[1], reverse=True)
    return [
        {
            "content": recalled_doc.page_content,
            "type": recalled_doc.metadata.get("type", SESSION_SUMMARY_TYPE),
        }
        for recalled_doc, final_score in rescored_results[:3]
        if final_score >= SESSION_MEMORY_SCORE_THRESHOLD
        and recalled_doc.metadata.get("type", SESSION_SUMMARY_TYPE)
        in {SESSION_SUMMARY_TYPE, LEGACY_AUTO_FACT_TYPE}
    ]


def save_study_note(
    study_notes_store,
    content: str,
    user_id: str,
    source_session_id: str = "",
    note_id: str | None = None,
    title: str = "",
    created_at: str = "",
    updated_at: str = "",
) -> str:
    """保存或覆盖一条跨 session 的研究笔记，并返回 note_id。"""
    note_id = note_id or f"note_{uuid4().hex}"
    now = datetime.now().isoformat()
    cleaned_content = content.strip()
    cleaned_title = title.strip() or cleaned_content[:30]
    note_created_at = created_at or now
    note_updated_at = updated_at or now

    # 研究笔记是用户资产，需要稳定的 note_id，便于后续更新和删除。
    # 这里直接用同一个 id 覆盖写，避免“先删再写”在中途失败时把旧数据删掉。
    note_doc = Document(
        page_content=cleaned_content,
        metadata={
            "note_id": note_id,
            "user_id": user_id,
            "content": cleaned_content,
            "title": cleaned_title,
            "created_at": note_created_at,
            "updated_at": note_updated_at,
            "source_session_id": source_session_id,
        },
    )
    study_notes_store.add_documents([note_doc], ids=[note_id])
    return note_id


def list_study_notes(
    study_notes_store,
    user_id: str,
) -> list[dict[str, str]]:
    """列出当前用户的全部研究笔记，供前端和工具层做管理操作。"""
    return _iter_user_note_records(study_notes_store, user_id)


def search_study_notes(
    study_notes_store,
    user_id: str,
    query: str,
    limit: int = STUDY_NOTES_RECALL_LIMIT,
    score_threshold: float = STUDY_NOTES_SCORE_THRESHOLD,
) -> list[dict[str, str]]:
    """按语义相关性检索当前用户的研究笔记。"""
    if not query.strip():
        return []
    # 这里不吞掉底层存储异常。
    # 如果向量库查询本身失败，应该让工具错误恢复层把它和“没有匹配结果”区分开。
    recall_results = study_notes_store.similarity_search_with_score(
        query,
        k=max(limit, 5),
        filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
    )

    matched_notes: list[dict[str, str]] = []
    for recalled_doc, similarity_score in recall_results:
        if similarity_score < score_threshold:
            continue
        matched_notes.append(
            {
                "note_id": str(recalled_doc.metadata.get("note_id", "")),
                "title": str(recalled_doc.metadata.get("title", "")),
                "content": recalled_doc.page_content,
                "updated_at": str(recalled_doc.metadata.get("updated_at", "")),
                "source_session_id": str(recalled_doc.metadata.get("source_session_id", "")),
            }
        )
    return matched_notes[:limit]


def update_study_note(
    study_notes_store,
    user_id: str,
    note_id: str,
    new_content: str,
    new_title: str = "",
) -> bool:
    """更新指定研究笔记的正文和标题，保持 note_id 不变。"""
    existing_note = next(
        (note for note in _iter_user_note_records(study_notes_store, user_id) if note["note_id"] == note_id),
        None,
    )
    if not existing_note:
        return False

    updated_content = new_content.strip()
    updated_title = new_title.strip() or updated_content[:30]
    save_study_note(
        study_notes_store,
        updated_content,
        user_id=user_id,
        source_session_id=existing_note.get("source_session_id", ""),
        note_id=note_id,
        title=updated_title,
        created_at=existing_note.get("created_at", ""),
        updated_at=datetime.now().isoformat(),
    )
    return True


def delete_study_note(
    study_notes_store,
    user_id: str,
    note_id: str,
) -> bool:
    """删除指定研究笔记。"""
    if not any(note["note_id"] == note_id for note in _iter_user_note_records(study_notes_store, user_id)):
        return False
    study_notes_store.delete(ids=[note_id])
    return True


def build_episodic_memory_prompt_block(memory_entries: list[dict[str, str]]) -> str:
    """把结构化 session summary 条目转成 prompt 片段。"""
    if not memory_entries:
        return ""

    # session memory 现在只负责当前会话的冷上下文续航，
    # 不再混入用户主动保存的 study notes。
    session_summaries = [
        entry
        for entry in memory_entries
        if entry["type"] in {SESSION_SUMMARY_TYPE, LEGACY_AUTO_FACT_TYPE}
    ]
    prompt_lines = ["【本会话的相关摘要】："]
    if session_summaries:
        prompt_lines.append("[会话摘要]")
        prompt_lines.extend(f"- {entry['content']}" for entry in session_summaries)
    prompt_lines.append("请在回答时参考以上背景。")
    return "\n".join(prompt_lines)


def build_study_notes_prompt_block(note_entries: list[dict[str, str]]) -> str:
    """把研究笔记召回结果转成单独的 prompt 片段。"""
    if not note_entries:
        return ""

    prompt_lines = ["【用户研究笔记】："]
    for note in note_entries:
        note_title = note.get("title", "").strip()
        note_prefix = f"[{note_title}] " if note_title else ""
        prompt_lines.append(f"- {note_prefix}{note['content']}")
    prompt_lines.append("这些内容是用户主动沉淀的研究笔记，回答相关问题时优先复用。")
    return "\n".join(prompt_lines)


def save_session_memory_entry(
    session_memory_store,
    content: str,
    user_id: str,
    session_id: str,
    entry_type: str = SESSION_SUMMARY_TYPE,
):
    """把一条 session memory 写入 Qdrant。"""
    # session memory 的主内容放在 page_content，
    # metadata 只保存后续过滤和排序需要的最小字段。
    memory_doc = Document(
        page_content=content,
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "type": entry_type,
            "timestamp": datetime.now().isoformat(),
        },
    )
    session_memory_store.add_documents([memory_doc])


def compress_conversation_window(
    conversation_messages: list[dict],
    session_memory_store,
    user_id: str,
    session_id: str,
    fast_llm,
    recovery_manager: ModelRecoveryManager | None = None,
) -> list[dict]:
    """
    消息窗口压缩：当消息总数超过 SESSION_WINDOW_SIZE 时，把最早的一轮对话
    压缩成摘要存入向量库，然后从消息列表里删掉这一轮。

    压缩逻辑分三步：
    1. 按轮次分组：每条 user 消息开启新一轮，该轮包含它之后直到下一条
       user 消息之前的所有消息，包括 assistant、tool 消息。
       例：[user, assistant(tool_calls), tool, assistant(最终回答)] 是一轮。
    2. 筛选完整轮次：只有轮次内存在"不带 tool_calls 的 assistant 消息"才算
       完整可压缩。这样可以避免删掉还没收口的工具调用序列。
    3. 整轮删除：把最早的一个完整轮次里的所有消息（user、assistant、tool）
       一起移除。这样 assistant(tool_calls) 和对应的 tool 消息永远作为整体
       一起删，不会留下孤立的 tool 消息导致下一轮 API 报 400。

    摘要文本只取 user/assistant 的可读内容，tool 消息不纳入摘要。
    """
    print(f"[Memory] 检查窗口压缩 | 总消息数={len(conversation_messages)}")

    # 第一步：按轮次分组。
    # 遇到 user 消息就把之前积累的消息存为一轮，然后开启新一轮。
    turns: list[list[dict]] = []
    current_turn: list[dict] = []
    for msg in conversation_messages:
        if msg["role"] == "user" and current_turn:
            turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)

    # 第二步：筛选完整轮次。
    # 轮次内有不带 tool_calls 的 assistant 消息，说明这一轮已经有最终回答，可以压缩。
    compressible_turns = [
        turn for turn in turns
        if any(
            m["role"] == "assistant" and not m.get("tool_calls")
            for m in turn
        )
    ]

    # 用完整轮次里的 user/assistant 消息数量判断是否达到压缩阈值。
    visible_conversation = [
        m for turn in compressible_turns for m in turn if m["role"] in ("user", "assistant")
    ]
    if (
        len(conversation_messages) <= SESSION_WINDOW_SIZE
        or len(visible_conversation) < SESSION_COMPRESS_BATCH_SIZE
    ):
        print("[Memory] 当前无需压缩窗口")
        return conversation_messages

    # 第三步：整轮删除。
    # 每次只处理最早的一个完整轮次，把这一轮的所有消息（含 tool）放入待删列表。
    turns_to_compress = compressible_turns[:1]
    messages_to_compress = [m for turn in turns_to_compress for m in turn]

    # 摘要文本只取 user/assistant 的可读内容，tool 消息不纳入摘要。
    dialogue_text = "\n".join(
        f"{'用户' if message['role'] == 'user' else 'AI'}: {str(message['content'])}"
        for message in messages_to_compress
        if message["role"] in ("user", "assistant") and message.get("content")
    )
    if not dialogue_text.strip():
        return conversation_messages

    summary_prompt = (
        "你在为长对话做窗口压缩，请把下面这段对话总结成可供后续检索的冷上下文。\n"
        "要求：\n"
        "1. 最多三句话。\n"
        "2. 只保留后续可能还会用到的事实、结论、用户要求或未解决问题。\n"
        "3. 不要复述寒暄、套话、无效客套。\n"
        "4. 只输出摘要本身，不要加标题、解释或项目符号。\n\n"
        "待压缩对话：\n"
        f"{dialogue_text}"
    )
    recovery_manager = recovery_manager or ModelRecoveryManager()
    print("[Memory] 开始生成窗口压缩摘要")
    summary_result = recovery_manager.invoke_text_model(
        fast_llm,
        summary_prompt,
        purpose="会话压缩摘要",
        fallback_value=dialogue_text[:200],
    )
    print(f"[Memory] 压缩摘要调用结束 | ok={summary_result.ok} | fallback={summary_result.used_fallback}")
    compressed_summary = str(getattr(summary_result.value, "content", summary_result.value) or "").strip()

    # 当前版本对写入长度做一次裁剪，防止模型输出过多内容塞回 memory。
    if compressed_summary:
        save_session_memory_entry(
            session_memory_store,
            compressed_summary[:200],
            user_id,
            session_id,
            entry_type=SESSION_SUMMARY_TYPE,
        )
        print(f"[Memory] 压缩写入 session memory: {compressed_summary[:60]}")

    # 用对象 id 标记要删的消息，避免误删内容相同的其他消息。
    compressed_message_ids = {id(message) for message in messages_to_compress}
    remaining_messages = [
        message
        for message in conversation_messages
        if id(message) not in compressed_message_ids
    ]
    print(f"[Memory] 窗口压缩：{len(conversation_messages)} → {len(remaining_messages)} 条消息")
    return remaining_messages


class UserPreferenceProfileStore:
    """管理跨 session 的用户偏好状态、偏好检测和确认流程。"""

    user_id: str  # 当前用户标识，用来定位用户自己的 profile 目录。
    profile_dir: Path  # 当前用户 profile 文件所在目录。
    state_file: Path  # 当前生效偏好状态文件，保存 profile 的最新快照。
    events_file: Path  # 偏好变更事件日志文件，按行追加历史操作。
    profile_state: dict  # 内存中的当前偏好状态快照，system prompt 读取它来注入 profile。

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.profile_dir = USER_PROFILE_ROOT / user_id
        self.state_file = self.profile_dir / "PROFILE_STATE.json"
        self.events_file = self.profile_dir / "PROFILE_EVENTS.jsonl"
        self.profile_state: dict = {}

    def load_state(self):
        """加载当前用户的偏好状态；状态文件不存在时保持空字典。"""
        self.profile_state = {}
        if not self.state_file.exists():
            return

        try:
            loaded_state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Profile] 读取状态失败，忽略损坏文件: {exc}")
            return

        # 这里只恢复“当前生效状态”，不会在启动时把事件日志整段重放。
        if isinstance(loaded_state, dict):
            self.profile_state = loaded_state

        slot_count = sum(
            len(scope_state)
            for scope_state in self.profile_state.values()
            if isinstance(scope_state, dict)
        )
        if slot_count:
            print(f"[Profile] 加载 {slot_count} 个偏好槽位（user={self.user_id}）")

    def should_analyze_preference(self, user_message: str) -> bool:
        """只在明显像长期偏好或偏好管理指令的输入上触发 fast_llm 分类。"""
        # 这里只做“是否值得触发分类”的候选筛选，不在规则层直接判断 add/update/delete。
        # 规则必须尽量用短语而不是单字，避免“不要删除”这类句子被单个“删”字误伤。
        candidate_hints = (
            LONG_TERM_PREFERENCE_HINTS
            + PROFILE_MANAGEMENT_HINTS
            + EXPLICIT_PREFERENCE_HINTS
        )
        return any(token in user_message for token in candidate_hints)

    def analyze_preference_candidate(
        self,
        user_message: str,
        fast_llm,
        recovery_manager: ModelRecoveryManager | None = None,
    ) -> dict | None:
        """调用带 schema 的 fast_llm 做结构化分类，再做归一化和业务校验。"""
        # 这一层只负责把自然语言转成结构化候选，
        # 真正要不要写入 profile，仍然由上层按置信度和敏感性决定。
        detection_prompt = (
            "你是一个用户偏好检测器。请判断下面这句话是否应写入跨 session 的长期偏好。\n"
            "请基于 schema 返回结构化结果，不要把这句话当成当前轮的临时要求。\n\n"
            "判定规则：\n"
            "1. 只有跨 session 仍然有价值的长期偏好才算 is_preference=true。\n"
            "2. 表达删除、清空、停止记录偏好的话，operation 用 delete 或 clear。\n"
            "3. scope/type/value 尽量标准化到系统支持的枚举和值。\n"
            "4. 如果不是长期偏好，is_preference=false，confidence=low。\n"
            "5. 遇到删除、清空、关闭记忆等高副作用操作时，照实标记 operation，后续程序会再确认。\n\n"
            f"用户原话：{user_message}"
        )
        recovery_manager = recovery_manager or ModelRecoveryManager()
        detection_result = recovery_manager.invoke_structured_model(
            fast_llm,
            PreferenceDetectionResult,
            detection_prompt,
            purpose="偏好检测",
        )
        if not detection_result.ok:
            print(f"[Profile] 偏好检测失败，忽略本轮候选: {detection_result.message}")
            return None

        parsed_result = detection_result.value.model_dump()
        if not parsed_result["is_preference"]:
            return {
                "is_preference": False,
                "confidence": "low",
                "reason": parsed_result.get("reason", "").strip(),
            }

        # 先把自由文本映射成系统内部接受的枚举值，再交给 Pydantic 做最终校验。
        normalized_data = {
            "is_preference": True,
            "scope": SCOPE_ALIASES.get(str(parsed_result.get("scope", "")).strip()),
            "type": TYPE_ALIASES.get(str(parsed_result.get("type", "")).strip()),
            "value": str(parsed_result.get("value", "")).strip(),
            "operation": OPERATION_ALIASES.get(str(parsed_result.get("operation", "")).strip().lower()),
            "confidence": CONFIDENCE_ALIASES.get(
                str(parsed_result.get("confidence", "")).strip().lower()
            ),
            "reason": str(parsed_result.get("reason", "")).strip() or "长期偏好候选",
            "sensitive": False,
        }

        # 删除和清空属于高副作用操作，后续必须进入确认流程。
        if normalized_data["operation"] in {"delete", "clear"}:
            normalized_data["sensitive"] = True

        try:
            validated_result = PreferenceDetectionResult.model_validate(normalized_data)
        except ValidationError:
            return None

        parsed_result = validated_result.model_dump()
        operation = parsed_result["operation"]
        # 不同 operation 对字段完整度的要求不同，
        # 这里做最后一层业务校验，避免半合法数据写入 profile。
        if operation in {"add", "update"}:
            if not parsed_result["scope"] or not parsed_result["type"] or not parsed_result["value"]:
                return None
        elif operation == "delete":
            if not parsed_result["scope"] or not parsed_result["type"]:
                return None
        elif operation == "clear":
            parsed_result["value"] = ""
        else:
            return None
        return parsed_result

    def build_prompt_block(self) -> str:
        """把当前生效的偏好状态转换成简短 prompt 块，供每轮全量注入。"""
        if not self.profile_state:
            return ""

        prompt_lines = ["【用户长期偏好与背景】（跨 session 持久）："]
        for scope in PROFILE_SCOPES:
            scope_state = self.profile_state.get(scope, {})
            if not isinstance(scope_state, dict) or not scope_state:
                continue

            # 这里按固定 scope/type 顺序输出，
            # 目的是让 prompt 块稳定、可预测，不随着字典顺序乱跳。
            prompt_lines.append(f"\n[{SCOPE_LABELS.get(scope, scope)}]")
            for preference_type in PROFILE_TYPES:
                record = scope_state.get(preference_type)
                if not isinstance(record, dict) or not record.get("value"):
                    continue
                prompt_lines.append(
                    f"- {TYPE_LABELS.get(preference_type, preference_type)}：{record['value']}"
                )
        return "\n".join(prompt_lines)

    def build_confirmation_message(self, preference: dict) -> str:
        """为中置信度或敏感操作生成程序层固定确认话术。"""
        if preference["operation"] == "clear":
            return "检测到你可能想清空长期偏好。要现在清空吗？请直接回复“是”或“否”。"

        if preference["operation"] == "delete":
            return (
                "检测到你可能想删除长期偏好："
                f"[{SCOPE_LABELS.get(preference['scope'], preference['scope'])}] "
                f"{TYPE_LABELS.get(preference['type'], preference['type'])}。"
                "要现在删除吗？请直接回复“是”或“否”。"
            )

        return (
            "我理解你可能希望把这条偏好记为长期设置："
            f"[{SCOPE_LABELS.get(preference['scope'], preference['scope'])}] "
            f"{TYPE_LABELS.get(preference['type'], preference['type'])} = {preference['value']}。"
            "要记住它吗？请直接回复“是”或“否”。"
        )

    def apply_preference_update(
        self,
        preference: dict,
        source_text: str,
        confirmed: bool = False,
    ) -> str:
        """按 scope + type 更新当前偏好状态，并把变更写入事件日志。"""
        now = datetime.now().isoformat()
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        # 校验 scope 和 type，不合法时直接返回错误，避免后续 KeyError。
        pref_scope = preference.get("scope", "")
        pref_type = preference.get("type", "")
        pref_op = preference.get("operation", "add")
        if pref_scope not in SCOPE_LABELS and pref_op != "clear":
            return f"保存失败：scope '{pref_scope}' 不合法，只能是 {list(SCOPE_LABELS.keys())}。"
        if pref_type not in TYPE_LABELS and pref_op not in {"clear", "delete"}:
            return f"保存失败：type '{pref_type}' 不合法，只能是 {list(TYPE_LABELS.keys())}。"

        # profile_state 只保存“当前生效值”，
        # 所以新增和更新都会直接覆盖同槽位旧值。
        if preference["operation"] == "clear":
            self.profile_state = {}
            result_message = "已清空长期偏好。"
        else:
            scope_state = self.profile_state.setdefault(preference["scope"], {})
            if preference["operation"] == "delete":
                if scope_state.pop(preference["type"], None) is not None:
                    result_message = (
                        f"已删除偏好：{SCOPE_LABELS[preference['scope']]} / "
                        f"{TYPE_LABELS[preference['type']]}。"
                    )
                else:
                    result_message = "未找到可删除的偏好。"
            else:
                preference_exists = preference["type"] in scope_state
                scope_state[preference["type"]] = {
                    "value": preference["value"],
                    "source_text": source_text,
                    "updated_at": now,
                    "confidence": preference["confidence"],
                }
                action_text = "已更新偏好" if preference_exists else "已保存偏好"
                result_message = (
                    f"{action_text}：{SCOPE_LABELS[preference['scope']]} / "
                    f"{TYPE_LABELS[preference['type']]} = {preference['value']}。"
                )

        # 先落当前状态文件，供下次启动时直接恢复。
        self.state_file.write_text(
            json.dumps(self.profile_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 再追加事件日志，保留后续审计和排查所需的历史轨迹。
        event_record = {
            "timestamp": now,
            "user_id": self.user_id,
            "source_text": source_text,
            "confirmed": confirmed,
            **preference,
        }
        with self.events_file.open("a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(event_record, ensure_ascii=False) + "\n")

        print(f"[Profile] {result_message}")
        return result_message

    def parse_confirmation_reply(self, user_message: str) -> bool | None:
        """把用户对确认问题的回答映射成 是 / 否 / 无法判断。"""
        normalized_message = user_message.strip().lower()
        if normalized_message in CONFIRM_TRUE_WORDS:
            return True
        if normalized_message in CONFIRM_FALSE_WORDS:
            return False
        return None


def get_user_preference_store(user_id: str) -> UserPreferenceProfileStore:
    """创建并初始化当前用户的偏好状态对象。"""
    preference_store = UserPreferenceProfileStore(user_id)
    preference_store.load_state()
    return preference_store


# 兼容旧命名，避免当前轮重构影响其他模块。
init_qdrant = init_vector_stores
get_memory_entries = retrieve_session_memory_entries
format_episodic_context = build_episodic_memory_prompt_block
save_fact = save_session_memory_entry
compress_window = compress_conversation_window
ProfileMemoryManager = UserPreferenceProfileStore
get_profile_manager = get_user_preference_store
