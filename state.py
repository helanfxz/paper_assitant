# state.py
# 定义所有 Agent 共享的图状态 SupervisorState。
# messages: 通过 add_messages reducer 自动累积对话历史
# next: Supervisor 写入，控制路由
# plan: 复杂任务时由 Supervisor 生成的步骤列表，空列表表示无计划
# plan_step: 当前执行到第几步
# step_results: 各步骤执行结果，最终由汇总节点合并成回答

from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next: str
    plan: list[str]           # ["ResearchAgent: 在s-lora.pdf中检索...", ...]
    plan_step: int            # 当前步骤索引
    step_results: list[str]   # 各步骤收集的结果
