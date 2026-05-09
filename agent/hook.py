"""Agent 执行生命周期 Hook。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class HookContext:
    """单轮执行上下文。"""

    iteration: int  # 当前是本轮问答里的第几次 model/tool 往返。
    messages: list[dict[str, Any]]  # 当前本轮工作消息列表。
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 当前这次模型提出的工具调用。
    tool_results: list[Any] = field(default_factory=list)  # 当前这次工具执行结果。
    final_content: str | None = None  # 当前轮最终回答。
    error: str | None = None  # 当前轮错误信息。


class Hook:
    """Hook 基类。"""

    def __init__(self, reraise: bool = False):
        self._reraise = reraise  # 为 True 时，Hook 异常直接向上抛出。

    def before_iteration(self, context: HookContext) -> None:
        """每次进入模型调用前执行。"""

    def before_execute_tools(self, context: HookContext) -> None:
        """工具执行前执行，可修改 context.tool_calls。"""

    def after_iteration(self, context: HookContext) -> None:
        """每次模型/工具往返完成后执行。"""

    def finalize_content(self, context: HookContext, content: str | None) -> str | None:
        """最终回答返回给上层前执行，可修改回答文本。"""
        return content


class CompositeHook(Hook):
    """顺序执行多个 Hook。"""

    def __init__(self, hooks: list[Hook]):
        super().__init__()
        self._hooks = list(hooks)  # 本轮真正参与执行的 Hook 列表。

    def _safe_call(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """按顺序调用 Hook，并对非关键 Hook 做异常隔离。"""
        for hook in self._hooks:
            if getattr(hook, "_reraise", False):
                getattr(hook, method_name)(*args, **kwargs)
                continue

            try:
                getattr(hook, method_name)(*args, **kwargs)
            except Exception as exc:
                print(f"[Hook] {type(hook).__name__}.{method_name} 执行失败: {exc}")

    def before_iteration(self, context: HookContext) -> None:
        self._safe_call("before_iteration", context)

    def before_execute_tools(self, context: HookContext) -> None:
        self._safe_call("before_execute_tools", context)

    def after_iteration(self, context: HookContext) -> None:
        self._safe_call("after_iteration", context)

    def finalize_content(self, context: HookContext, content: str | None) -> str | None:
        # 最终回答是管道式处理，前一个 Hook 的输出作为后一个 Hook 的输入。
        for hook in self._hooks:
            content = hook.finalize_content(context, content)
        return content


class ConfirmHook(Hook):
    """管理“延后确认”的写入动作。"""

    def __init__(self, pending_actions: list[dict[str, Any]]):
        super().__init__()
        self.pending_actions = pending_actions  # 当前会话的待确认动作队列。
        self._current_turn_ids: list[str] = []  # 当前这轮新加入的待确认动作 id。

    def start_turn(self) -> None:
        """每轮 ask 开始前重置本轮新增动作记录。"""
        self._current_turn_ids = []

    def queue_preference_action(
        self,
        preference: dict[str, Any],
        source_text: str,
        summary: str,
    ) -> None:
        """把长期偏好写入动作转成待确认项。"""
        approval_id = self._build_approval_id()
        self.pending_actions.append(
            {
                "approval_id": approval_id,
                "kind": "preference",
                "tool_name": "save_preference",
                "tool_args": {
                    "scope": preference.get("scope", ""),
                    "type": preference.get("type", ""),
                    "value": preference.get("value", ""),
                    "operation": preference.get("operation", "add"),
                    "source_text": source_text,
                },
                "tool_call_id": f"pref_{approval_id.lower()}",
                "summary": summary,
            }
        )
        self._current_turn_ids.append(approval_id)

    def before_execute_tools(self, context: HookContext) -> None:
        """拦截未来需要确认后再执行的写入型工具。"""
        executable_tool_calls: list[dict[str, Any]] = []

        # 这里只拦截确认型工具，其他工具仍然正常执行，保证主回答完整性。
        for tool_call in context.tool_calls:
            if not self._needs_confirmation(tool_call):
                executable_tool_calls.append(tool_call)
                continue

            approval_id = self._build_approval_id()
            self.pending_actions.append(
                {
                    "approval_id": approval_id,
                    "kind": "tool",
                    "tool_name": tool_call["name"],
                    "tool_args": dict(tool_call.get("args", {})),
                    "tool_call_id": str(tool_call.get("id", "")).strip() or f"tool_{approval_id.lower()}",
                    "summary": self._build_tool_summary(tool_call),
                }
            )
            self._current_turn_ids.append(approval_id)

        context.tool_calls = executable_tool_calls

    def finalize_content(self, context: HookContext, content: str | None) -> str | None:
        """在本轮回答末尾追加待确认提示。"""
        if not self._current_turn_ids:
            return content

        new_actions = [
            action
            for action in self.pending_actions
            if action.get("approval_id") in self._current_turn_ids
        ]
        if not new_actions:
            return content

        lines = [content or "", "", "待确认操作："]
        for action in new_actions:
            lines.append(f"- {action['summary']}")
        lines.append("请在界面下方使用“同意 / 拒绝 / 补充意见”继续处理。")
        return "\n".join(lines).strip()

    def _needs_confirmation(self, tool_call: dict[str, Any]) -> bool:
        """判断某个工具调用是否应转成待确认动作。"""
        tool_name = str(tool_call.get("name", "")).strip()
        tool_args = tool_call.get("args", {}) or {}
        if tool_name == "save_preference":
            return True
        if tool_name == "save_todo":
            return True
        return False

    def _build_tool_summary(self, tool_call: dict[str, Any]) -> str:
        """把工具调用参数转成前端可读的确认摘要。"""
        tool_name = str(tool_call.get("name", "")).strip()
        tool_args = tool_call.get("args", {}) or {}
        if tool_name == "save_preference":
            scope = str(tool_args.get("scope", "")).strip() or "global"
            pref_type = str(tool_args.get("type", "")).strip() or "unknown"
            value = str(tool_args.get("value", "")).strip()
            operation = str(tool_args.get("operation", "add")).strip() or "add"
            if operation == "clear":
                return "清空长期偏好"
            if operation == "delete":
                return f"删除长期偏好：{scope} / {pref_type}"
            return f"保存长期偏好：{scope} / {pref_type} = {value or '未提供'}"
        if tool_name == "save_todo":
            title = str(tool_args.get("title", "")).strip()
            detail = str(tool_args.get("detail", "")).strip()
            status = str(tool_args.get("status", "todo")).strip() or "todo"
            time_kind = str(tool_args.get("time_kind", "none")).strip() or "none"
            time_desc = f" | time={time_kind}" if time_kind != "none" else ""
            subtasks = tool_args.get("subtasks") or []
            lines = [f"保存会话 todo：{title or detail or '未命名任务'} | status={status}{time_desc}"]
            if detail:
                lines.append(f"  详情：{detail}")
            if subtasks:
                lines.append("  子任务：")
                for index, subtask in enumerate(subtasks, 1):
                    lines.append(f"  {index}. {subtask}")
            return "\n".join(lines)
        return f"执行写入操作：{tool_name}"

    def _build_approval_id(self) -> str:
        """生成短 approval_id，便于前端展示和排查。"""
        return f"A-{uuid4().hex[:6].upper()}"
