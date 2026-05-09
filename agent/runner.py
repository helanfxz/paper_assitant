"""单轮 model -> tool -> model 执行循环。"""

from __future__ import annotations

from typing import Any, Callable

from agent.error_recovery import ToolRecoveryManager
from agent.hook import Hook, HookContext
from agent.provider import CONTEXT_TOO_LONG_ERROR, LLMResponse, Provider
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

CONTINUE_MESSAGE = "输出被截断，请直接从中断处继续，不要重复已有内容。"
FALLBACK_REPLY = "抱歉，当前请求处理失败，请稍后重试。"
MAX_CONTINUE_RETRIES = 3
MAX_AGENT_ITERATIONS = 8  # 单轮问答中最多允许 model/tool 往返次数，避免模型持续调用工具导致请求卡住。


class AgentRunner:
    """负责单轮推理闭环，不负责 session 持久化。"""

    provider: Provider
    final_provider: Provider
    tools_by_name: dict[str, BaseTool]
    tool_recovery: ToolRecoveryManager
    hook: Hook

    def __init__(
        self,
        provider: Provider,
        tools_by_name: dict[str, BaseTool],
        tool_recovery: ToolRecoveryManager,
        hook: Hook | None = None,
        final_provider: Provider | None = None,
    ):
        self.provider = provider
        self.final_provider = final_provider or provider
        self.tools_by_name = tools_by_name
        self.tool_recovery = tool_recovery
        self.hook = hook or Hook()

    def _build_llm_messages(
        self,
        system_prompt: str,
        turn_messages: list[dict[str, Any]],
    ) -> list[Any]:
        """把内部消息列表转换成 LangChain 消息对象。"""
        llm_messages: list[Any] = [SystemMessage(content=system_prompt)]

        for message in turn_messages:
            role = message.get("role", "")
            if role == "user":
                llm_messages.append(HumanMessage(content=message.get("content", "")))
                continue

            if role == "tool":
                llm_messages.append(
                    ToolMessage(
                        content=str(message.get("content", "")),
                        tool_call_id=message.get("tool_call_id", ""),
                        name=message.get("name", ""),
                    )
                )
                continue

            ai_kwargs: dict[str, Any] = {
                "content": message.get("content", "")
                if isinstance(message.get("content", ""), str)
                else "",
            }
            if message.get("tool_calls"):
                ai_kwargs["tool_calls"] = list(message.get("tool_calls", []))
            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                ai_kwargs["additional_kwargs"] = {
                    "reasoning_content": reasoning_content,
                }
            llm_messages.append(AIMessage(**ai_kwargs))

        return llm_messages

    def _run_tools(
        self,
        tool_calls: list[dict[str, Any]],
        turn_messages: list[dict[str, Any]],
    ) -> list[Any]:
        """执行本轮工具调用，并把 tool message 写回本轮消息列表。"""
        read_tools_with_recovery = {
            "list_documents",
            "search_pdf",
            "recall_memory",
            "list_notes",
            "search_notes",
            "get_stats",
        }
        write_tools_with_recovery = {"save_note", "update_note", "delete_note"}

        tool_results: list[Any] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_impl = self.tools_by_name.get(tool_name)
            print(f"  [Tool] {tool_name} | 参数: {tool_args}")

            tool_call_result = None
            if tool_name in read_tools_with_recovery:
                tool_call_result = self.tool_recovery.invoke_read_tool(
                    tool_name,
                    tool_impl,
                    tool_args,
                )
            elif tool_name in write_tools_with_recovery:
                tool_call_result = self.tool_recovery.invoke_write_tool(
                    tool_name,
                    tool_impl,
                    tool_args,
                )

            if tool_call_result is not None:
                tool_result = tool_call_result.content
                if not tool_call_result.ok:
                    print(f"  [Tool] {tool_name} 失败: {tool_call_result.message}")
            else:
                try:
                    tool_result = (
                        tool_impl.invoke(tool_args)
                        if tool_impl
                        else f"未知工具: {tool_name}"
                    )
                except Exception as exc:
                    tool_result = f"工具执行失败: {exc}"
                    print(f"  [Tool] {tool_name} 失败: {exc}")

            turn_messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_name,
                }
            )
            tool_results.append(tool_result)

        return tool_results

    def _call_model(
        self,
        llm_messages: list[Any],
        on_content_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """调用主模型；有增量回调时走流式，否则走普通调用。"""
        print(
            f"[Runner] 开始模型调用 | mode={'stream' if on_content_delta is not None else 'sync'} | "
            f"messages={len(llm_messages)}"
        )
        if on_content_delta is not None:
            response = self.provider.chat_stream_with_retry(
                llm_messages,
                on_content_delta=on_content_delta,
            )
        else:
            response = self.provider.chat_with_retry(llm_messages)
        print(
            f"[Runner] 模型调用结束 | finish_reason={response.finish_reason} | "
            f"tool_calls={len(response.tool_calls)}"
        )
        return response

    def run(
        self,
        system_prompt: str,
        turn_messages: list[dict[str, Any]],
        on_content_delta: Callable[[str], None] | None = None,
    ) -> str:
        """完成一整轮 model -> tool -> model 执行循环。"""
        continuation_count = 0
        iteration = 0
        final_answer = FALLBACK_REPLY

        context = HookContext(
            iteration=iteration,
            messages=turn_messages,
        )

        while True:
            iteration += 1
            if iteration > MAX_AGENT_ITERATIONS:
                context.error = "超过最大工具调用轮数"
                print(f"[Runner] 超过最大迭代次数 {MAX_AGENT_ITERATIONS}，强制收口")
                turn_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "工具调用次数已达到上限。不要再调用工具，"
                            "请只基于当前已有上下文和工具结果给出最终回答。"
                        ),
                        "hidden": True,
                    }
                )
                llm_messages = self._build_llm_messages(system_prompt, turn_messages)
                model_response = self.final_provider.chat_with_retry(llm_messages)
                if model_response.finish_reason == "error":
                    print(f"[Runner] 最终收口模型调用失败: {model_response.error}")
                final_answer = (
                    model_response.content
                    or "本轮工具调用次数已达到上限，但无法生成最终回答。"
                )
                context.final_content = final_answer
                turn_messages.append({"role": "assistant", "content": final_answer})
                break

            context.iteration = iteration
            print(f"[Runner] 进入迭代 {iteration}")
            self.hook.before_iteration(context)

            llm_messages = self._build_llm_messages(system_prompt, turn_messages)
            model_response = self._call_model(
                llm_messages,
                on_content_delta=on_content_delta,
            )

            if model_response.finish_reason == "error":
                if model_response.error_type == CONTEXT_TOO_LONG_ERROR:
                    context.error = "上下文过长"
                    return "上下文过长，请尝试缩短问题或开启新会话。"

                context.error = model_response.error or "模型调用失败"
                print(f"[Provider] 主模型调用失败: {context.error}")
                return FALLBACK_REPLY

            if model_response.finish_reason == "length":
                continuation_count += 1
                if continuation_count <= MAX_CONTINUE_RETRIES:
                    assistant_message = {"role": "assistant", "content": model_response.content or ""}
                    if model_response.reasoning_content:
                        assistant_message["reasoning_content"] = model_response.reasoning_content
                    turn_messages.append(assistant_message)
                    turn_messages.append({"role": "user", "content": CONTINUE_MESSAGE})
                    continue

                final_answer = model_response.content or FALLBACK_REPLY
                context.final_content = final_answer
                break

            continuation_count = 0

            if not model_response.has_tool_calls:
                print("[Runner] 本轮无工具调用，准备收口回答")
                final_answer = model_response.content or FALLBACK_REPLY
                final_answer = self.hook.finalize_content(context, final_answer)

                assistant_message = {"role": "assistant", "content": final_answer}
                if model_response.reasoning_content:
                    assistant_message["reasoning_content"] = model_response.reasoning_content
                turn_messages.append(assistant_message)
                context.final_content = final_answer
                self.hook.after_iteration(context)
                break

            raw_tool_calls = [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "args": tool_call.arguments,
                }
                for tool_call in model_response.tool_calls
            ]
            context.tool_calls = list(raw_tool_calls)
            print(f"[Runner] 本轮工具调用数: {len(context.tool_calls)}")

            self.hook.before_execute_tools(context)
            if not context.tool_calls:
                print("[Runner] 工具调用已被 Hook 全部拦截，直接收口当前回答")
                final_answer = model_response.content or ""
                final_answer = self.hook.finalize_content(context, final_answer)
                if not final_answer:
                    final_answer = FALLBACK_REPLY
                assistant_message = {"role": "assistant", "content": final_answer}
                if model_response.reasoning_content:
                    assistant_message["reasoning_content"] = model_response.reasoning_content
                turn_messages.append(assistant_message)
                context.final_content = final_answer
                self.hook.after_iteration(context)
                break

            turn_messages.append(
                {
                    "role": "assistant",
                    "content": model_response.content or "",
                    "tool_calls": list(context.tool_calls),
                    "reasoning_content": model_response.reasoning_content,
                }
            )
            tool_results = self._run_tools(context.tool_calls, turn_messages)
            print(f"[Runner] 工具执行完成，结果数: {len(tool_results)}")
            context.tool_results = tool_results
            self.hook.after_iteration(context)

        return final_answer
