"""统一的 API 调用恢复层。

当前只处理模型 / API 相关失败，不处理工具执行失败。
这一层统一回答几件事：
1. 这次调用是否成功
2. 如果失败，属于哪种类型
3. 是否已经使用降级结果

上层模块只需要关心业务兜底，不再各自手写重试与分类逻辑。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

RATE_LIMIT_ERROR = "rate_limit"
TRANSIENT_NETWORK_ERROR = "transient_network"
CONTEXT_TOO_LONG_ERROR = "context_too_long"
STRUCTURED_OUTPUT_INVALID_ERROR = "structured_output_invalid"
PROVIDER_ERROR = "provider_error"
UNKNOWN_ERROR = "unknown"
INVALID_ARGS_ERROR = "invalid_args"
TRANSIENT_TOOL_ERROR = "transient_error"
DEPENDENCY_UNAVAILABLE_ERROR = "dependency_unavailable"
EMPTY_RESULT_ERROR = "empty_result"
PERMISSION_DENIED_ERROR = "permission_denied"
TOOL_EXECUTION_ERROR = "execution_error"
TOOL_NOT_FOUND_ERROR = "tool_not_found"
TOOL_TYPE_READ = "read"
TOOL_TYPE_WRITE = "write"
TOOL_TYPE_HIGH_RISK = "high_risk"


@dataclass(slots=True)
class ApiCallResult:
    """统一描述一次模型调用结果。"""

    ok: bool  # 当前调用是否成功拿到可用结果。
    value: Any = None  # 成功结果，或失败时的降级值。
    error_type: str = ""  # 失败类型；成功时为空。
    message: str = ""  # 供日志打印的简短说明。
    used_fallback: bool = False  # 失败后是否使用了兜底值。


@dataclass(slots=True)
class ToolCallResult:
    """统一描述一次工具调用结果。"""

    ok: bool  # 工具是否成功返回可用结果。
    content: str  # 工具返回的原始结果文本，渲染给模型前会按需要包装成 observation。
    error_type: str = ""  # 失败类型；成功时为空。
    message: str = ""  # 程序层日志使用的简短说明。
    should_stop: bool = False  # 这次失败是否应该直接终止当前整轮对话。
    used_fallback: bool = False  # 这次结果是否来自降级路径，而不是完整成功路径。
    recoverable: bool = False  # 模型或程序是否还有机会修正这次失败。
    suggested_next_action: str = ""  # 给模型和 runner 的下一步建议。
    tool_type: str = ""  # read / write / high_risk，便于 runner 做统一控制。
    attempts: int = 1  # 实际尝试执行工具的次数。

    def to_observation(self, tool_name: str) -> str:
        """把结构化结果渲染成给 LLM 看的 tool message。"""
        if self.ok and not self.error_type and not self.used_fallback and not self.suggested_next_action:
            return self.content

        lines = [
            "TOOL_OBSERVATION:",
            f"tool: {tool_name}",
            f"ok: {str(self.ok).lower()}",
        ]
        if self.tool_type:
            lines.append(f"tool_type: {self.tool_type}")
        if self.error_type:
            lines.append(f"error_type: {self.error_type}")
        lines.append(f"recoverable: {str(self.recoverable).lower()}")
        lines.append(f"used_fallback: {str(self.used_fallback).lower()}")
        lines.append(f"should_stop: {str(self.should_stop).lower()}")
        if self.message:
            lines.append(f"message: {self.message}")
        if self.suggested_next_action:
            lines.append(f"suggested_next_action: {self.suggested_next_action}")
        if self.content:
            lines.append("content:")
            lines.append(self.content)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """描述单个工具的恢复策略，避免策略散落在 runner 里。"""

    tool_type: str = TOOL_TYPE_READ  # 工具类型决定默认失败处理方式。
    required_args: tuple[str, ...] = ()  # 执行前必须存在且非空的参数。
    retryable: bool = False  # 是否允许程序层自动重试。
    max_retries: int = 0  # transient_error 下最多自动重试次数。
    stop_on_failure: bool = False  # 失败后是否直接收口，不再让模型继续兜圈子。


class ModelRecoveryManager:
    """统一处理主模型、fast_llm 和 rerank_llm 的 API 恢复策略。"""

    max_retries: int  # 限流 / 网络抖动时允许的最大重试次数。
    backoff_base: float  # 指数退避起始秒数。
    backoff_max: float  # 指数退避最大秒数。
    structured_validation_retries: int  # 结构化输出校验失败时允许的补救次数。

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        structured_validation_retries: int = 1,
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.structured_validation_retries = structured_validation_retries

    def _backoff_seconds(self, attempt: int) -> float:
        """为可重试错误生成指数退避等待时间。"""
        return min(self.backoff_base * (2**attempt), self.backoff_max) + random.uniform(0, 1)

    def _classify_exception(self, exc: Exception) -> str:
        """把原始异常归一为项目当前使用的几类错误。"""
        message = str(exc).lower()

        if any(token in message for token in ("context length", "token limit", "too long", "overlong")):
            return CONTEXT_TOO_LONG_ERROR
        if any(token in message for token in ("rate limit", "429", "too many requests", "overloaded")):
            return RATE_LIMIT_ERROR
        if any(token in message for token in ("connection", "timeout", "network", "socket")):
            return TRANSIENT_NETWORK_ERROR
        if any(token in message for token in ("api", "server", "provider", "service unavailable", "500", "502", "503", "504")):
            return PROVIDER_ERROR
        return UNKNOWN_ERROR

    def invoke_text_model(
        self,
        llm: Any,
        payload: Any,
        purpose: str,
        fallback_value: Any = None,
    ) -> ApiCallResult:
        """统一执行普通文本/消息调用，并处理重试与失败降级。"""
        for attempt in range(self.max_retries + 1):
            try:
                return ApiCallResult(ok=True, value=llm.invoke(payload))
            except Exception as exc:
                error_type = self._classify_exception(exc)
                if error_type == CONTEXT_TOO_LONG_ERROR:
                    return ApiCallResult(
                        ok=False,
                        value=fallback_value,
                        error_type=error_type,
                        message=f"{purpose} 失败：上下文过长",
                        used_fallback=fallback_value is not None,
                    )

                if error_type in {RATE_LIMIT_ERROR, TRANSIENT_NETWORK_ERROR, PROVIDER_ERROR} and attempt < self.max_retries:
                    wait_seconds = self._backoff_seconds(attempt)
                    print(
                        f"[Recovery] {purpose} 重试 {attempt + 1}/{self.max_retries}，"
                        f"等待 {wait_seconds:.1f}s: {exc}"
                    )
                    time.sleep(wait_seconds)
                    continue

                return ApiCallResult(
                    ok=False,
                    value=fallback_value,
                    error_type=error_type,
                    message=f"{purpose} 失败: {exc}",
                    used_fallback=fallback_value is not None,
                )

        return ApiCallResult(
            ok=False,
            value=fallback_value,
            error_type=UNKNOWN_ERROR,
            message=f"{purpose} 失败：超过重试次数",
            used_fallback=fallback_value is not None,
        )

    def invoke_structured_model(
        self,
        llm: Any,
        schema: type,
        prompt: str,
        purpose: str,
        fallback_value: Any = None,
    ) -> ApiCallResult:
        """统一执行结构化输出调用，并处理校验失败后的补救逻辑。"""
        validation_failures = 0

        for attempt in range(self.max_retries + 1):
            try:
                # 这里统一优先尝试 function-calling 风格的结构化输出，
                # 如果当前模型包装不支持，再退回默认的结构化输出接口。
                try:
                    structured_llm = llm.with_structured_output(schema, method="function_calling")
                except TypeError:
                    structured_llm = llm.with_structured_output(schema)

                structured_result = structured_llm.invoke(prompt)

                # 调试：打印LLM返回的原始结果
                if structured_result is None:
                    print(f"[Recovery] {purpose} LLM返回了None，可能是模型拒绝回答或结构化输出失败")
                    raise ValidationError.from_exception_data(
                        "value_error",
                        [{"type": "model_type", "input": None, "msg": "LLM returned None"}]
                    )

                validated_result = schema.model_validate(structured_result)
                return ApiCallResult(ok=True, value=validated_result)
            except ValidationError as exc:
                validation_failures += 1
                if validation_failures <= self.structured_validation_retries:
                    print(
                        f"[Recovery] {purpose} 结构化校验失败，补救 {validation_failures}/"
                        f"{self.structured_validation_retries}: {exc}"
                    )
                    continue

                return ApiCallResult(
                    ok=False,
                    value=fallback_value,
                    error_type=STRUCTURED_OUTPUT_INVALID_ERROR,
                    message=f"{purpose} 失败：结构化输出不合法",
                    used_fallback=fallback_value is not None,
                )
            except Exception as exc:
                error_type = self._classify_exception(exc)
                if error_type == CONTEXT_TOO_LONG_ERROR:
                    return ApiCallResult(
                        ok=False,
                        value=fallback_value,
                        error_type=error_type,
                        message=f"{purpose} 失败：上下文过长",
                        used_fallback=fallback_value is not None,
                    )

                if error_type in {RATE_LIMIT_ERROR, TRANSIENT_NETWORK_ERROR, PROVIDER_ERROR} and attempt < self.max_retries:
                    wait_seconds = self._backoff_seconds(attempt)
                    print(
                        f"[Recovery] {purpose} 重试 {attempt + 1}/{self.max_retries}，"
                        f"等待 {wait_seconds:.1f}s: {exc}"
                    )
                    time.sleep(wait_seconds)
                    continue

                return ApiCallResult(
                    ok=False,
                    value=fallback_value,
                    error_type=error_type,
                    message=f"{purpose} 失败: {exc}",
                    used_fallback=fallback_value is not None,
                )

        return ApiCallResult(
            ok=False,
            value=fallback_value,
            error_type=UNKNOWN_ERROR,
            message=f"{purpose} 失败：超过重试次数",
            used_fallback=fallback_value is not None,
        )


class ToolRecoveryManager:
    """统一处理工具执行失败，并向 runner 输出可控制的结构化结果。"""

    def __init__(self):
        self.default_read_policy = ToolPolicy(
            tool_type=TOOL_TYPE_READ,
            retryable=True,
            max_retries=1,
            stop_on_failure=False,
        )
        self.default_write_policy = ToolPolicy(
            tool_type=TOOL_TYPE_WRITE,
            retryable=False,
            max_retries=0,
            stop_on_failure=True,
        )
        self.tool_policies: dict[str, ToolPolicy] = {
            "load_skill": ToolPolicy(TOOL_TYPE_READ, required_args=("name",), retryable=False),
            "list_documents": self.default_read_policy,
            "search_pdf": ToolPolicy(TOOL_TYPE_READ, required_args=("query",), retryable=True, max_retries=1),
            "recall_memory": ToolPolicy(TOOL_TYPE_READ, required_args=("query",), retryable=True, max_retries=1),
            "list_notes": self.default_read_policy,
            "search_notes": ToolPolicy(TOOL_TYPE_READ, required_args=("query",), retryable=True, max_retries=1),
            "get_stats": self.default_read_policy,
            "list_todos": self.default_read_policy,
            "save_note": ToolPolicy(TOOL_TYPE_WRITE, required_args=("content",), stop_on_failure=True),
            "update_note": ToolPolicy(TOOL_TYPE_WRITE, required_args=("note_id", "new_content"), stop_on_failure=True),
            "delete_note": ToolPolicy(TOOL_TYPE_HIGH_RISK, required_args=("note_id",), stop_on_failure=True),
            "save_preference": ToolPolicy(TOOL_TYPE_WRITE, required_args=("value",), stop_on_failure=True),
            "save_todo": ToolPolicy(TOOL_TYPE_WRITE, required_args=("title",), stop_on_failure=True),
            "update_todo_status": ToolPolicy(
                TOOL_TYPE_WRITE,
                required_args=("todo_id", "status"),
                stop_on_failure=True,
            ),
            "delete_todo": ToolPolicy(TOOL_TYPE_HIGH_RISK, required_args=("todo_id",), stop_on_failure=True),
        }

    def _policy_for(self, tool_name: str, fallback_type: str = TOOL_TYPE_READ) -> ToolPolicy:
        """读取工具策略，未知工具按调用方给出的类型使用默认策略。"""
        if tool_name in self.tool_policies:
            return self.tool_policies[tool_name]
        if fallback_type in {TOOL_TYPE_WRITE, TOOL_TYPE_HIGH_RISK}:
            return self.default_write_policy
        return self.default_read_policy

    def _missing_tool_result(self, tool_name: str, policy: ToolPolicy) -> ToolCallResult:
        """统一生成“工具不存在”的结果，避免各分支重复拼接。"""
        return ToolCallResult(
            ok=False,
            content=f"{tool_name} 不存在。",
            error_type=TOOL_NOT_FOUND_ERROR,
            message=f"{tool_name} 不存在",
            should_stop=False,
            recoverable=True,
            suggested_next_action="请改用系统 prompt 中列出的可用工具，不要继续调用不存在的工具。",
            tool_type=policy.tool_type,
        )

    def _validate_args(
        self,
        tool_name: str,
        tool_args: Any,
        policy: ToolPolicy,
    ) -> ToolCallResult | None:
        """执行前做轻量参数校验；参数错误不应自动重试。"""
        if not isinstance(tool_args, dict):
            return ToolCallResult(
                ok=False,
                content=f"{tool_name} 的参数必须是对象。",
                error_type=INVALID_ARGS_ERROR,
                message=f"{tool_name} 参数不是 dict",
                recoverable=True,
                suggested_next_action="请按工具 schema 重新生成参数后再调用。",
                tool_type=policy.tool_type,
            )

        for arg_name in policy.required_args:
            value = tool_args.get(arg_name)
            is_empty_string = isinstance(value, str) and not value.strip()
            is_empty_collection = isinstance(value, (list, tuple, dict)) and not value
            if value is None or is_empty_string or is_empty_collection:
                return ToolCallResult(
                    ok=False,
                    content=f"{tool_name} 的必填参数 {arg_name} 不能为空。",
                    error_type=INVALID_ARGS_ERROR,
                    message=f"{tool_name} 缺少必填参数 {arg_name}",
                    recoverable=True,
                    suggested_next_action=f"请补全 {arg_name} 后重新调用 {tool_name}。",
                    tool_type=policy.tool_type,
                )
        return None

    def _classify_tool_exception(self, exc: Exception) -> str:
        """把工具异常归类成 runner 可消费的错误类型。"""
        message = str(exc).lower()
        if any(token in message for token in ("validation", "field required", "missing", "invalid args", "bad request")):
            return INVALID_ARGS_ERROR
        if any(token in message for token in ("permission", "unauthorized", "forbidden", "401", "403")):
            return PERMISSION_DENIED_ERROR
        if any(token in message for token in ("timeout", "connection", "network", "socket", "429", "502", "503", "504")):
            return TRANSIENT_TOOL_ERROR
        if any(token in message for token in ("qdrant", "collection", "index required", "not found", "rerank", "database")):
            return DEPENDENCY_UNAVAILABLE_ERROR
        return TOOL_EXECUTION_ERROR

    def _should_retry(self, policy: ToolPolicy, error_type: str, attempt: int) -> bool:
        """只有读工具的临时错误适合自动重试，参数错误和写入错误不重试。"""
        return (
            policy.tool_type == TOOL_TYPE_READ
            and policy.retryable
            and error_type == TRANSIENT_TOOL_ERROR
            and attempt < policy.max_retries
        )

    def _backoff_seconds(self, attempt: int) -> float:
        """工具重试使用短退避，避免单轮问答被拖得过久。"""
        return min(0.5 * (2**attempt), 2.0) + random.uniform(0, 0.3)

    def _success_result(self, tool_name: str, tool_output: Any, policy: ToolPolicy, attempts: int) -> ToolCallResult:
        """把成功输出转成统一结果；空结果是状态，不是异常。"""
        if isinstance(tool_output, ToolCallResult):
            if not tool_output.tool_type:
                tool_output.tool_type = policy.tool_type
            tool_output.attempts = attempts
            return tool_output

        content = str(tool_output)
        empty_markers = ("未找到相关内容", "暂无文档", "暂无笔记", "暂无 todo")
        if any(marker in content for marker in empty_markers):
            return ToolCallResult(
                ok=True,
                content=content,
                error_type=EMPTY_RESULT_ERROR,
                message=f"{tool_name} 成功执行但没有结果",
                recoverable=True,
                suggested_next_action="可以换一个更具体的查询继续尝试，或基于当前没有结果的事实回答。",
                tool_type=policy.tool_type,
                attempts=attempts,
            )
        return ToolCallResult(ok=True, content=content, tool_type=policy.tool_type, attempts=attempts)

    def _failure_result(
        self,
        tool_name: str,
        exc: Exception,
        error_type: str,
        policy: ToolPolicy,
        attempts: int,
    ) -> ToolCallResult:
        """把最终失败转成统一结果，明确告诉 runner 是否需要停止。"""
        recoverable = error_type in {INVALID_ARGS_ERROR, TRANSIENT_TOOL_ERROR, TOOL_NOT_FOUND_ERROR}
        if error_type == INVALID_ARGS_ERROR:
            next_action = "请修正工具参数后再调用，不要使用同一组错误参数重试。"
        elif error_type == TRANSIENT_TOOL_ERROR:
            next_action = "当前工具遇到临时错误；如果已有足够信息，请直接回答，否则稍后再试。"
        elif error_type == DEPENDENCY_UNAVAILABLE_ERROR:
            next_action = "当前工具依赖不可用；请不要继续调用同类工具，改用已有上下文收口。"
        elif error_type == PERMISSION_DENIED_ERROR:
            next_action = "权限不足或用户拒绝；不要重试该操作。"
        else:
            next_action = "不要反复调用同一工具；请基于已有上下文说明失败原因。"

        return ToolCallResult(
            ok=False,
            content=f"{tool_name} 执行失败：{exc}",
            error_type=error_type,
            message=f"{tool_name} 执行失败: {exc}",
            should_stop=policy.stop_on_failure or error_type in {PERMISSION_DENIED_ERROR},
            recoverable=recoverable,
            suggested_next_action=next_action,
            tool_type=policy.tool_type,
            attempts=attempts,
        )

    def invoke_tool(
        self,
        tool_name: str,
        tool_impl: Any,
        tool_args: Any,
        fallback_type: str = TOOL_TYPE_READ,
    ) -> ToolCallResult:
        """按工具策略执行工具，并返回 runner 可消费的结构化结果。"""
        policy = self._policy_for(tool_name, fallback_type=fallback_type)
        if tool_impl is None:
            return self._missing_tool_result(tool_name, policy)

        invalid_args = self._validate_args(tool_name, tool_args, policy)
        if invalid_args is not None:
            return invalid_args

        attempts = 0
        for attempt in range(policy.max_retries + 1):
            attempts = attempt + 1
            try:
                tool_output = tool_impl.invoke(tool_args)
                return self._success_result(tool_name, tool_output, policy, attempts)
            except Exception as exc:
                error_type = self._classify_tool_exception(exc)
                if self._should_retry(policy, error_type, attempt):
                    wait_seconds = self._backoff_seconds(attempt)
                    print(
                        f"[ToolRecovery] {tool_name} 重试 {attempt + 1}/{policy.max_retries}，"
                        f"等待 {wait_seconds:.1f}s: {exc}"
                    )
                    time.sleep(wait_seconds)
                    continue
                return self._failure_result(tool_name, exc, error_type, policy, attempts)

        return ToolCallResult(
            ok=False,
            content=f"{tool_name} 执行失败：超过重试次数。",
            error_type=UNKNOWN_ERROR,
            message=f"{tool_name} 执行失败：超过重试次数",
            should_stop=policy.stop_on_failure,
            recoverable=False,
            suggested_next_action="不要继续调用该工具，请基于已有信息收口。",
            tool_type=policy.tool_type,
            attempts=attempts,
        )

    def invoke_read_tool(
        self,
        tool_name: str,
        tool_impl: Any,
        tool_args: dict[str, Any],
    ) -> ToolCallResult:
        """兼容旧调用：执行查询类工具。"""
        return self.invoke_tool(tool_name, tool_impl, tool_args, fallback_type=TOOL_TYPE_READ)

    def invoke_write_tool(
        self,
        tool_name: str,
        tool_impl: Any,
        tool_args: dict[str, Any],
    ) -> ToolCallResult:
        """兼容旧调用：执行写入类工具。"""
        return self.invoke_tool(tool_name, tool_impl, tool_args, fallback_type=TOOL_TYPE_WRITE)
