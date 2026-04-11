# graph/builder.py
# 负责组装主图：Supervisor + 三个专家 Agent + Synthesizer。
#
# 图结构：
#   简单模式：START → supervisor → Agent → supervisor → FINISH → END
#   计划模式：START → supervisor → Agent → supervisor → Agent → ... → synthesizer → END
#
# synthesizer 节点：在计划全部执行完后，将各步骤结果汇总成最终回答。

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from state import SupervisorState
from graph.supervisor import build_supervisor_node
from agents.research import make_research_agent
from agents.memory_agent import make_memory_agent
from agents.general import make_general_agent


def build_graph(llm, fast_llm, pdf_store, memory_store, user_id: str, session_id: str,
                stats: dict, loaded_docs: list, checkpointer):

    research_agent = make_research_agent(llm, fast_llm, pdf_store, loaded_docs)
    memory_agent = make_memory_agent(llm, memory_store, user_id, session_id, stats)
    general_agent = make_general_agent(llm, stats)

    def _invoke_agent(agent, state: SupervisorState, agent_name: str) -> dict:
        """通用 Agent 调用：处理计划模式下的任务注入。"""
        messages = list(state["messages"])

        # 计划模式：把当前步骤的任务描述注入为 SystemMessage，让 Agent 明确目标
        plan = state.get("plan", [])
        plan_step = state.get("plan_step", 0)
        if plan and plan_step < len(plan):
            task = plan[plan_step]
            # 提取任务描述（去掉 "AgentName: " 前缀）
            task_desc = task.split(":", 1)[-1].strip() if ":" in task else task
            messages = [SystemMessage(content=f"【当前任务】：{task_desc}")] + messages

        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        print(f"[{agent_name}] 回答: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        return {"messages": [AIMessage(content=answer, name=agent_name)]}

    def research_node(state: SupervisorState) -> dict:
        print("[Supervisor] → ResearchAgent")
        return _invoke_agent(research_agent, state, "ResearchAgent")

    def memory_node(state: SupervisorState) -> dict:
        print("[Supervisor] → MemoryAgent")
        return _invoke_agent(memory_agent, state, "MemoryAgent")

    def general_node(state: SupervisorState) -> dict:
        print("[Supervisor] → GeneralAgent")
        return _invoke_agent(general_agent, state, "GeneralAgent")

    def synthesizer_node(state: SupervisorState) -> dict:
        """计划全部执行完后，汇总各步骤结果生成最终回答。"""
        print("[Synthesizer] 汇总各步骤结果...")
        step_results = state.get("step_results", [])
        plan = state.get("plan", [])
        user_question = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )

        steps_summary = "\n\n".join([
            f"步骤{i+1}（{plan[i] if i < len(plan) else ''}）：\n{result}"
            for i, result in enumerate(step_results)
        ])

        prompt = (
            f"用户问题：{user_question}\n\n"
            f"以下是按计划分步检索到的内容：\n\n{steps_summary}\n\n"
            f"请基于以上所有内容，给出完整、有条理的回答。"
        )
        answer = llm.invoke(prompt).content
        print(f"[Synthesizer] 汇总完成，回答长度: {len(answer)} 字")

        # 清空计划状态，为下一轮对话做准备
        return {
            "messages": [AIMessage(content=answer, name="Synthesizer")],
            "plan": [],
            "plan_step": 0,
            "step_results": [],
        }

    # ── 图组装 ────────────────────────────────────────────────
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", build_supervisor_node(fast_llm))
    workflow.add_node("ResearchAgent", research_node)
    workflow.add_node("MemoryAgent", memory_node)
    workflow.add_node("GeneralAgent", general_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "ResearchAgent": "ResearchAgent",
            "MemoryAgent":   "MemoryAgent",
            "GeneralAgent":  "GeneralAgent",
            "synthesizer":   "synthesizer",
            "FINISH":        END,
        }
    )

    # 所有 Agent 执行完都回到 Supervisor
    workflow.add_edge("ResearchAgent", "supervisor")
    workflow.add_edge("MemoryAgent",   "supervisor")
    workflow.add_edge("GeneralAgent",  "supervisor")
    # Synthesizer 完成后结束
    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=checkpointer)
