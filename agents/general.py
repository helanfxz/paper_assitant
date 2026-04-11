# agents/general.py
# 定义 GeneralAgent 的工具和系统提示。
# get_stats：查询当前会话的学习统计信息（时长、文档数、提问数、笔记数）。
# 也负责处理不需要工具的日常闲聊。

from datetime import datetime
from langchain_core.tools import tool
from agents.base import build_sub_agent

SYSTEM_PROMPT = (
    "你是友好的通用助手。"
    "可以使用 get_stats 查询学习统计，也可以直接回答日常问题，无需工具。"
)


def make_general_agent(llm, stats: dict):
    @tool
    def get_stats() -> str:
        """获取当前会话的学习统计信息。"""
        duration = (datetime.now() - stats["session_start"]).seconds
        return (
            f"会话时长: {duration}秒 | 加载文档: {stats['docs_loaded']} | "
            f"提问次数: {stats['questions_asked']} | 笔记数量: {stats['notes_added']}"
        )

    return build_sub_agent(llm, [get_stats], SYSTEM_PROMPT, name="GeneralAgent", max_tool_calls=1)
