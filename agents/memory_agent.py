# agents/memory_agent.py
# 定义 MemoryAgent 的工具和系统提示。
# recall_memory：从 Qdrant 检索该 session 的历史笔记和自动提取的事实，支持类型过滤。
# add_note：将用户提供的内容手动保存为笔记存入 Qdrant。

from langchain_core.tools import tool
from agents.base import build_sub_agent
from memory.store import save_fact

SYSTEM_PROMPT = (
    "你是用户的个人记忆管理专家。\n"
    "使用 recall_memory 查询本会话的历史笔记和重要事实，"
    "使用 add_note 保存用户明确要求记录的新笔记。\n"
    "每次回答只需调用一次工具，调用后直接基于结果作答。"
)


def make_memory_agent(llm, memory_store, user_id: str, session_id: str, stats: dict):
    @tool
    def recall_memory(query: str, memory_type: str = "all") -> str:
        """从本会话的历史笔记和自动提取的事实中检索相关记忆。
        memory_type 可选：'note'（手动笔记）| 'auto_fact'（自动提取）| 'all'（全部）
        """
        print(f"  [MemoryAgent][TOOL] recall_memory 开始，query: {query}, type: {memory_type}")
        must_filters = [
            {"key": "user_id", "match": {"value": user_id}},
            {"key": "session_id", "match": {"value": session_id}},
        ]
        if memory_type != "all":
            must_filters.append({"key": "type", "match": {"value": memory_type}})
        try:
            docs = memory_store.similarity_search(
                query, k=5, filter={"must": must_filters}
            )
        except Exception as e:
            print(f"  [MemoryAgent][TOOL] 检索失败: {e}")
            docs = []

        if not docs:
            print(f"  [MemoryAgent][TOOL] 未找到相关记忆")
            return "未找到相关记忆。"

        results = []
        for d in docs:
            ts = d.metadata.get("timestamp", "")
            mtype = d.metadata.get("type", "note")
            results.append(f"[{mtype}{' | ' + ts[:10] if ts else ''}]: {d.page_content}")
        print(f"  [MemoryAgent][TOOL] 找到 {len(results)} 条记忆")
        return "\n\n".join(results)

    @tool
    def add_note(content: str) -> str:
        """将用户提供的内容保存为学习笔记到本会话记忆库。"""
        print(f"  [MemoryAgent][TOOL] add_note: {content[:50]}")
        save_fact(memory_store, content, user_id, session_id, fact_type="note")
        stats["notes_added"] += 1
        return f"笔记已保存：{content[:50]}..."

    return build_sub_agent(llm, [recall_memory, add_note], SYSTEM_PROMPT, name="MemoryAgent", max_tool_calls=1)
