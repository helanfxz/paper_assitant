# agents/base.py
# 通用子图工厂，用于构建标准 ReAct 子图（agent ⇄ tools 循环）。
# 每个专家 Agent 通过调用 build_sub_agent() 生成，传入各自的工具列表、
# 系统提示、名称和最大工具调用次数。
# max_tool_calls 双重保障：系统提示层面主动约束 + should_continue 计数强制截断。

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, ToolMessage


def build_sub_agent(llm, tools: list, system_prompt: str, name: str = "Agent", max_tool_calls: int = 2):
    """
    构建一个标准 ReAct 子图。
    结构：START → agent ⇄ tools → END
    max_tool_calls: 工具调用次数上限，超过后强制结束，防止 Agent 反复查询。
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # 在系统提示中明确调用次数限制，让 LLM 主动收敛
    constrained_prompt = (
        f"{system_prompt}\n\n"
        f"【工具调用限制】：每次回答最多调用工具 {max_tool_calls} 次，"
        f"调用完毕后必须立即基于已有结果给出回答，不得继续调用。"
    )
    sys_msg = SystemMessage(content=constrained_prompt)

    def agent_node(state: MessagesState):
        print(f"  [{name}] 调用 LLM 中...")
        # GLM 要求 SystemMessage 只能在最开头。
        # 将 state 中所有 SystemMessage（压缩摘要、记忆注入）合并进 agent 系统提示，
        # 对话历史只保留 Human/AI/Tool 消息。
        extra = "\n\n".join(m.content for m in state["messages"] if isinstance(m, SystemMessage))
        merged_sys = SystemMessage(
            content=constrained_prompt + ("\n\n" + extra if extra else "")
        )
        conv_msgs = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        response = llm_with_tools.invoke([merged_sys] + conv_msgs)
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"  [{name}] 决策: 调用工具 {tc['name']} | 参数: {tc['args']}")
        else:
            print(f"  [{name}] 决策: 生成最终回答")
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        # 兜底计数：统计已执行的工具调用次数
        tool_call_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        if hasattr(last, "tool_calls") and last.tool_calls:
            if tool_call_count >= max_tool_calls:
                print(f"  [{name}] 已达工具调用上限({max_tool_calls})，强制结束")
                return END
            print(f"  [{name}] → 执行工具节点 (第{tool_call_count + 1}次)")
            return "tools"
        print(f"  [{name}] → 回答完成，返回主图")
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()
