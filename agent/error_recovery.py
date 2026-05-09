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
TOOL_EXECUTION_ERROR = "tool_execution_error"
TOOL_NOT_FOUND_ERROR = "tool_not_found"


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
    content: str  # 供模型继续观察的工具结果文本。
    error_type: str = ""  # 失败类型；成功时为空。
    message: str = ""  # 程序层日志使用的简短说明。
    should_stop: bool = False  # 这次失败是否应该直接终止当前整轮对话。
    used_fallback: bool = False  # 这次结果是否来自降级路径，而不是完整成功路径。


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
    """统一处理工具执行失败的外层策略。

    原则：
    1. 查询类工具失败后，优先让整轮继续执行
    2. 写入类工具失败后，优先保守返回明确错误
    3. 工具内部已经做过降级时，这里只负责兜住最终异常
    """

    def _missing_tool_result(self, tool_name: str) -> ToolCallResult:
        """统一生成“工具不存在”的结果，避免各分支重复拼接。"""
        return ToolCallResult(
            ok=False,
            content=f"[ToolError] {tool_name} 不存在。",
            error_type=TOOL_NOT_FOUND_ERROR,
            message=f"{tool_name} 不存在",
        )

    def _invoke_tool(
        self,
        tool_name: str,
        tool_impl: Any,
        tool_args: dict[str, Any],
    ) -> ToolCallResult:
        """执行一次工具调用，并把底层异常统一转成结构化结果。"""
        if tool_impl is None:
            return self._missing_tool_result(tool_name)

        try:
            tool_output = tool_impl.invoke(tool_args)
            return ToolCallResult(ok=True, content=str(tool_output))
        except Exception as exc:
            return ToolCallResult(
                ok=False,
                content=f"[ToolError] {tool_name} 执行失败：{exc}",
                error_type=TOOL_EXECUTION_ERROR,
                message=f"{tool_name} 执行失败: {exc}",
            )

    def invoke_read_tool(
        self,
        tool_name: str,
        tool_impl: Any,
        tool_args: dict[str, Any],
    ) -> ToolCallResult:
        """执行查询类工具，并把异常变成结构化的不中断结果。"""
        return self._invoke_tool(tool_name, tool_impl, tool_args)

    def invoke_write_tool(
        self,
        tool_name: str,
        tool_impl: Any,
        tool_args: dict[str, Any],
    ) -> ToolCallResult:
        """执行写入类工具。

        写入类工具默认不自动重试，避免重复写入或写到一半的中间状态。
        """
        return self._invoke_tool(tool_name, tool_impl, tool_args)

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
