# graph/supervisor.py
# Supervisor 路由节点，同时承担 Plan-and-Act 的调度职责。
#
# 两种工作模式：
# 1. 无计划模式（plan 为空）：
#    - 最后一条消息来自 Agent → FINISH
#    - 最后一条消息来自用户 → LLM 判断简单/复杂：
#        简单 → 直接路由到某个 Agent
#        复杂 → 生成 plan 存入 state，路由到 plan[0]
#
# 2. 计划执行模式（plan 不为空）：
#    - 刚收到 Agent 结果 → 收集结果，plan_step+1
#        还有步骤 → 路由到 plan[plan_step]
#        全部完成 → 路由到 synthesizer
#    - 尚未开始 → 路由到 plan[0]

import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import SupervisorState
from document.registry import format_doc_list

AGENT_NAMES = ("ResearchAgent", "MemoryAgent", "GeneralAgent")


def _last_agent_msg(state: SupervisorState):
    """返回最后一条来自专家 Agent 的 AIMessage，否则返回 None。"""
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and getattr(last, "name", None) in AGENT_NAMES:
        return last
    return None


def _last_user_msg(state: SupervisorState) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )


def build_supervisor_node(fast_llm):

    def supervisor_node(state: SupervisorState) -> dict:
        plan = state.get("plan", [])
        plan_step = state.get("plan_step", 0)
        step_results = state.get("step_results", [])
        last_agent = _last_agent_msg(state)

        # ── 计划执行模式 ──────────────────────────────────────
        if plan:
            if last_agent:
                # 收集本步结果，推进步骤
                new_results = step_results + [last_agent.content]
                new_step = plan_step + 1
                print(f"[Supervisor] 计划步骤 {plan_step + 1}/{len(plan)} 完成")

                if new_step < len(plan):
                    # 还有步骤，解析下一步的目标 Agent
                    next_agent = _parse_plan_step(plan[new_step])
                    print(f"[Supervisor] 执行计划步骤 {new_step + 1}: {plan[new_step]}")
                    return {"next": next_agent, "plan_step": new_step, "step_results": new_results}
                else:
                    # 全部完成，进入汇总
                    print(f"[Supervisor] 计划全部完成，进入汇总")
                    return {"next": "synthesizer", "plan_step": new_step, "step_results": new_results}
            else:
                # 计划已生成但尚未开始执行
                next_agent = _parse_plan_step(plan[0])
                print(f"[Supervisor] 开始执行计划步骤 1: {plan[0]}")
                return {"next": next_agent}

        # ── 无计划模式 ────────────────────────────────────────
        if last_agent:
            print(f"[Supervisor] 路由决策：FINISH（专家已回答）")
            return {"next": "FINISH"}

        user_msg = _last_user_msg(state)
        doc_info = format_doc_list()

        prompt = (
            f"你是主控路由节点。根据用户问题决定处理方式。\n\n"
            f"【知识库文档】：\n{doc_info}\n\n"
            f"【用户问题】：{user_msg}\n\n"
            f"【判断规则】：\n"
            f"如果问题需要查询多篇文档、对比分析、或多步骤处理，输出 JSON 计划。\n"
            f"计划中每个步骤必须包含：Agent名称 + 冒号 + 针对该步骤的具体检索关键词（从用户问题中提取）。\n"
            f"例如用户问'对比LoRAMoE和AlphaLoRA的方法'，应输出：\n"
            f'  {{"plan": ["ResearchAgent: LoRAMoE 核心方法 混合专家", "ResearchAgent: AlphaLoRA 核心方法 参数分配"]}}\n\n'
            f"如果是简单的单步问题，只输出一个词：\n"
            f"  GeneralAgent：问候、闲聊、不涉及文档的问题\n"
            f"  ResearchAgent：查询单篇文档内容、学术概念\n"
            f"  MemoryAgent：查询或保存笔记\n\n"
            f"只输出一个词或一个 JSON，不得有其他内容：\n"
        )

        raw = fast_llm.invoke(prompt).content.strip()
        print(f"[Supervisor] LLM 原始输出: {raw[:80]}")

        # 尝试解析为计划
        parsed_plan = _try_parse_plan(raw)
        if parsed_plan:
            print(f"[Supervisor] 生成计划，共 {len(parsed_plan)} 步：")
            for i, step in enumerate(parsed_plan):
                print(f"  步骤{i+1}: {step}")
            first_agent = _parse_plan_step(parsed_plan[0])
            return {
                "next": first_agent,
                "plan": parsed_plan,
                "plan_step": 0,
                "step_results": [],
            }

        # 简单路由
        next_agent = "GeneralAgent"
        for agent in ["ResearchAgent", "MemoryAgent", "GeneralAgent"]:
            if agent in raw:
                next_agent = agent
                break
        print(f"[Supervisor] 路由决策：{next_agent}")
        return {"next": next_agent, "plan": [], "plan_step": 0, "step_results": []}

    return supervisor_node


def _try_parse_plan(raw: str) -> list[str]:
    """尝试从 LLM 输出中解析 JSON 计划，失败返回空列表。"""
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1:
            return []
        data = json.loads(raw[start:end])
        plan = data.get("plan", [])
        if isinstance(plan, list) and len(plan) > 1:
            return [str(s) for s in plan]
    except Exception:
        pass
    return []


def _parse_plan_step(step: str) -> str:
    """从计划步骤字符串中提取目标 Agent 名称。
    格式示例：'ResearchAgent: 在 s-lora.pdf 中检索显存优化'
    """
    for agent in AGENT_NAMES:
        if agent in step:
            return agent
    return "ResearchAgent"  # 默认兜底
