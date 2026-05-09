"""模型调用抽象层。

这个文件提供一套很小的 provider 接口，让 runner 不再直接依赖
LangChain 的响应细节。当前实现只包装项目里现有的 LangChain 模型对象。
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

RATE_LIMIT_ERROR = "rate_limit"
TRANSIENT_NETWORK_ERROR = "transient_network"
CONTEXT_TOO_LONG_ERROR = "context_too_long"
PROVIDER_ERROR = "provider_error"
UNKNOWN_ERROR = "unknown"


@dataclass(slots=True)
class ToolCallRequest:
    """统一后的工具调用结构。"""

    id: str  # 模型生成的工具调用标识，用于和后续 tool result 对齐。
    name: str  # 要调用的工具名。
    arguments: dict[str, Any]  # 工具参数，统一规范成字典。


@dataclass(slots=True)
class LLMResponse:
    """统一后的模型响应结构。"""

    content: str | None  # 最终可见文本内容；工具调用轮次时可以为空。
    tool_calls: list[ToolCallRequest] = field(default_factory=list)  # 模型本轮请求调用的工具列表。
    finish_reason: str = "stop"  # stop / tool_calls / length / error。
    usage: dict[str, int] = field(default_factory=dict)  # 统一后的 token 使用统计。
    error: str | None = None  # 结构化错误文本；正常返回时为空。
    error_type: str | None = None  # 统一后的错误类型。
    retry_after: float | None = None  # 供应商给出的建议等待时间。
    should_retry: bool | None = None  # 这次错误是否值得继续重试。
    reasoning_content: str | None = None  # thinking 模式下后续请求需要原样回传的推理内容。

    @property
    def has_tool_calls(self) -> bool:
        """判断当前响应是否要求 runner 执行工具。"""
        return bool(self.tool_calls)


class Provider(ABC):
    """当前项目的最小模型调用接口。"""

    @abstractmethod
    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行一次普通模型调用，并返回统一响应。"""

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_content_delta: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行一次流式模型调用，并返回最终统一响应。"""

    @abstractmethod
    def chat_with_retry(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行带重试的普通模型调用，并返回统一响应。"""

    @abstractmethod
    def chat_stream_with_retry(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_content_delta: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行带重试的流式模型调用，并返回最终统一响应。"""


class ModelProvider(Provider):
    """把当前 LangChain 模型对象包装成稳定接口。"""

    model: Any  # 当前项目传入的 LangChain 模型，或已经 bind_tools 后的模型实例。
    max_retries: int  # 普通瞬时错误最多重试次数。

    def __init__(self, model: Any, max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行一次模型调用，并把返回结果转成统一结构。"""
        model_instance = self._prepare_model_instance(tools, temperature, max_tokens)
        result = model_instance.invoke(messages)
        return self._to_response(result)

    def chat_stream(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_content_delta: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行流式模型调用，并在结束后统一收口最终结果。"""
        model_instance = self._prepare_model_instance(tools, temperature, max_tokens)
        merged_chunk: Any | None = None

        # 逐个消费模型返回的流式分片：文本增量发给上层，完整状态留到最后统一收口。
        for chunk in model_instance.stream(messages):
            text_delta = self._extract_text(getattr(chunk, "content", None))
            if text_delta and on_content_delta is not None:
                on_content_delta(text_delta)

            if merged_chunk is None:
                merged_chunk = chunk
            else:
                merged_chunk = merged_chunk + chunk

        if merged_chunk is None:
            return LLMResponse(content="", finish_reason="stop")

        return self._to_response(merged_chunk)

    def chat_with_retry(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行带重试的普通模型调用。"""
        return self._run_with_retry(
            lambda: self.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    def chat_stream_with_retry(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_content_delta: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """执行带重试的流式模型调用。"""
        return self._run_with_retry(
            lambda: self.chat_stream(
                messages,
                tools=tools,
                on_content_delta=on_content_delta,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    def _run_with_retry(self, call: Callable[[], LLMResponse]) -> LLMResponse:
        """统一处理主模型 API 的瞬时错误重试。"""
        last_error: LLMResponse | None = None

        # 重试只负责供应商级瞬时错误；长度截断、空回复等由 runner 层继续处理。
        for attempt in range(1, self.max_retries + 1):
            try:
                response = call()
            except Exception as exc:
                response = self._exception_to_response(exc)

            if response.finish_reason != "error":
                return response

            last_error = response
            if not response.should_retry or response.error_type == CONTEXT_TOO_LONG_ERROR:
                return response

            if attempt >= self.max_retries:
                return response

            wait_seconds = response.retry_after or self._backoff_seconds(attempt)
            print(
                f"[Provider] 模型调用失败，{wait_seconds:.1f}s 后重试 "
                f"({attempt}/{self.max_retries})：{response.error or '未知错误'}"
            )
            time.sleep(wait_seconds)

        return last_error or LLMResponse(
            content="模型调用失败。",
            finish_reason="error",
            error="模型调用失败。",
            error_type=UNKNOWN_ERROR,
            should_retry=False,
        )

    def _prepare_model_instance(
        self,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Any:
        """为当前调用准备模型实例，同时避免修改共享模型对象。"""
        model_instance = self.model

        # 这里先支持显式传入 tools，后面 runner 可以逐步统一改走 provider。
        if tools and hasattr(model_instance, "bind_tools"):
            model_instance = model_instance.bind_tools(tools)

        bind_kwargs: dict[str, Any] = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if bind_kwargs and hasattr(model_instance, "bind"):
            model_instance = model_instance.bind(**bind_kwargs)

        return model_instance

    def _to_response(self, result: Any) -> LLMResponse:
        """把 LangChain 返回对象转换成项目内部响应结构。"""
        content = self._extract_text(getattr(result, "content", None))
        tool_calls = self._parse_tool_calls(getattr(result, "tool_calls", None))
        finish_reason = self._parse_finish_reason(result, tool_calls)
        usage = self._parse_usage(result)
        reasoning_content = self._parse_reasoning_content(result)
        return LLMResponse(
            content=content or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def _exception_to_response(self, exc: Exception) -> LLMResponse:
        """把底层异常规整成统一错误响应，供重试逻辑使用。"""
        error_text = str(exc) or exc.__class__.__name__
        error_type = self._classify_exception(exc)
        retry_after = self._extract_retry_after(error_text)
        should_retry = error_type in {RATE_LIMIT_ERROR, TRANSIENT_NETWORK_ERROR, PROVIDER_ERROR}
        return LLMResponse(
            content=None,
            finish_reason="error",
            error=error_text,
            error_type=error_type,
            retry_after=retry_after,
            should_retry=should_retry,
        )

    @staticmethod
    def _classify_exception(exc: Exception) -> str:
        """按异常文本做当前项目所需的最小错误分类。"""
        message = str(exc).lower()
        if any(
            token in message
            for token in (
                "invalid_request_error",
                "must be passed back to the api",
                "messages with role 'tool'",
            )
        ):
            return UNKNOWN_ERROR
        if any(token in message for token in ("context length", "token limit", "too long", "overlong")):
            return CONTEXT_TOO_LONG_ERROR
        if any(token in message for token in ("rate limit", "429", "too many requests", "overloaded")):
            return RATE_LIMIT_ERROR
        if any(token in message for token in ("connection", "timeout", "network", "socket")):
            return TRANSIENT_NETWORK_ERROR
        if any(token in message for token in ("api", "server", "provider", "service unavailable", "500", "502", "503", "504")):
            return PROVIDER_ERROR
        return UNKNOWN_ERROR

    @staticmethod
    def _extract_retry_after(message: str) -> float | None:
        """从错误文本里提取供应商给出的重试等待时间。"""
        lowered = message.lower()
        patterns = (
            r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            value = float(match.group(1))
            unit = (match.group(2) or "s").lower()
            if unit in {"ms", "millisecond", "milliseconds"}:
                return value / 1000.0
            if unit in {"m", "min", "minutes"}:
                return value * 60.0
            return value
        return None

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """返回当前重试轮次对应的固定退避时间。"""
        return float(2 ** (attempt - 1))

    @staticmethod
    def _extract_text(content: Any) -> str:
        """把常见的 LangChain 内容结构压平成普通文本。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []

            # 流式 chunk 和多模态内容都可能是列表；这里只抽取可见文本部分。
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _parse_reasoning_content(result: Any) -> str | None:
        """提取需要在 thinking 模式下原样回传的 reasoning_content。"""
        direct_reasoning = getattr(result, "reasoning_content", None)
        if isinstance(direct_reasoning, str) and direct_reasoning.strip():
            return direct_reasoning

        additional_kwargs = getattr(result, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            raw_reasoning = additional_kwargs.get("reasoning_content")
            if isinstance(raw_reasoning, str) and raw_reasoning.strip():
                return raw_reasoning

        response_metadata = getattr(result, "response_metadata", None)
        if isinstance(response_metadata, dict):
            raw_reasoning = response_metadata.get("reasoning_content")
            if isinstance(raw_reasoning, str) and raw_reasoning.strip():
                return raw_reasoning
        return None

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCallRequest]:
        """把 LangChain 的 tool call 结构规整成稳定格式。"""
        if not isinstance(raw_tool_calls, list):
            return []

        parsed: list[ToolCallRequest] = []

        # 逐个规整 tool call，避免上层再分辨 args 是字典还是 JSON 字符串。
        for index, tool_call in enumerate(raw_tool_calls):
            if not isinstance(tool_call, dict):
                continue
            arguments = tool_call.get("args", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            parsed.append(
                ToolCallRequest(
                    id=str(tool_call.get("id") or f"tool_call_{index}"),
                    name=str(tool_call.get("name") or ""),
                    arguments=arguments,
                )
            )

        return parsed

    @staticmethod
    def _parse_finish_reason(result: Any, tool_calls: list[ToolCallRequest]) -> str:
        """从 LangChain 元数据里推断统一的结束原因。"""
        metadata = getattr(result, "response_metadata", None)
        if isinstance(metadata, dict):
            finish_reason = metadata.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                return finish_reason

        if tool_calls:
            return "tool_calls"
        return "stop"

    @staticmethod
    def _parse_usage(result: Any) -> dict[str, int]:
        """统一常见的 LangChain/OpenAI token 统计字段。"""
        usage = getattr(result, "usage_metadata", None)
        if not isinstance(usage, dict):
            return {}

        normalized: dict[str, int] = {}
        field_map = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        }

        # 统一 token 统计字段名，避免上层直接耦合 LangChain 的 usage_metadata 命名。
        for source_key, target_key in field_map.items():
            value = usage.get(source_key)
            if value is None:
                continue
            try:
                normalized[target_key] = int(value)
            except (TypeError, ValueError):
                continue

        return normalized
