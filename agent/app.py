"""AgentLoop：外层编排层。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import BaseTool
from langchain_qdrant import QdrantVectorStore

from agent.config import get_embeddings, get_fast_llm, get_llm, get_reranker
from agent.context import SystemContextBuilder
from agent.document import ensure_document_registry_from_vector_store, load_document
from agent.error_recovery import ModelRecoveryManager, ToolRecoveryManager
from agent.hook import ConfirmHook
from agent.memory import (
    UserPreferenceProfileStore as PreferenceStore,
    compress_conversation_window,
    delete_study_note,
    get_user_preference_store,
    init_vector_stores,
    list_study_notes,
    migrate_legacy_notes,
    save_study_note,
    update_study_note,
)
from agent.provider import ModelProvider
from agent.runner import AgentRunner
from agent.session import Session, SessionManager, get_sessions
from agent.skill import SkillLoader
from agent.tools import build_runtime_tools
from lexical_index import PersistentLexicalIndex


class AgentLoop:
    """项目主入口，负责一轮问答的编排与会话状态管理。"""

    user_id: str  # 当前用户标识。
    session_id: str  # 当前会话标识。
    primary_llm: Any  # 主模型实例。
    fast_llm: Any  # 轻量模型实例。
    rerank_llm: Any  # 检索重排模型实例。
    recovery: ModelRecoveryManager  # 结构化输出与轻量模型恢复层。
    tool_recovery: ToolRecoveryManager  # 工具恢复层。
    pdf_store: QdrantVectorStore  # 共享 PDF 知识库。
    lexical_index: PersistentLexicalIndex  # 持久化 BM25 索引。
    session_memory_store: QdrantVectorStore  # 当前会话摘要向量库。
    study_notes_store: QdrantVectorStore  # 跨会话研究笔记向量库。
    preference: PreferenceStore  # 当前用户的长期偏好存储。
    session_stats: dict[str, Any]  # 当前会话运行统计。
    loaded_document_names: list[str]  # 当前会话已加载文档名。
    is_first_user_turn: bool  # 是否仍处于首轮用户输入。
    pending_actions: list[dict[str, Any]]  # 待确认动作队列。
    confirm_hook: ConfirmHook  # 确认型工具拦截 Hook。
    tools: list[BaseTool]  # 当前轮可用工具。
    tool_llm: Any  # 已绑定工具后的主模型。
    provider: ModelProvider  # 主模型 provider。
    tools_by_name: dict[str, BaseTool]  # 工具名到实现的映射。
    session: Session  # 当前会话对象。
    sessions: SessionManager  # 会话管理器。
    context_builder: SystemContextBuilder  # System prompt 构建器。
    runner: AgentRunner  # 单轮执行循环。

    def __init__(self, user_id: str = "default_user", session_id: str | None = None):
        self.user_id = user_id
        self.sessions = get_sessions()
        self.session = self.sessions.get_or_create(session_id, user_id)
        self.session_id = self.session.session_id
        if session_id:
            print(f"[Session] 恢复会话: {self.session_id}")

        self.primary_llm = get_llm()
        self.fast_llm = get_fast_llm()
        self.rerank_llm = get_reranker()
        self.recovery = ModelRecoveryManager()
        self.tool_recovery = ToolRecoveryManager()
        self.lexical_index = PersistentLexicalIndex()
        self.skill_loader = SkillLoader()
        embeddings = get_embeddings()

        _, self.pdf_store, self.session_memory_store, self.study_notes_store = init_vector_stores(embeddings)
        ensure_document_registry_from_vector_store(self.pdf_store)
        migrate_legacy_notes(self.session_memory_store, self.study_notes_store, user_id)
        self.preference = get_user_preference_store(user_id)

        self.session_stats = {
            "session_start": datetime.now(),
            "docs_loaded": 0,
            "questions_asked": 0,
            "notes_added": 0,
        }
        self.loaded_document_names = []
        self.is_first_user_turn = self.session.is_new()
        self.pending_actions = list(self.session.runtime.pending_actions)
        if self._prune_expired_todos():
            self._save_session()

        self.tools = build_runtime_tools(
            fast_llm=self.fast_llm,
            rerank_llm=self.rerank_llm,
            recovery_manager=self.recovery,
            pdf_store=self.pdf_store,
            lexical_index=self.lexical_index,
            session_memory_store=self.session_memory_store,
            study_notes_store=self.study_notes_store,
            preference_store=self.preference,
            user_id=self.user_id,
            session_id=self.session_id,
            session_stats=self.session_stats,
            runtime=self.session.runtime,
            skill_loader=self.skill_loader,
        )
        self.tool_llm = self.primary_llm.bind_tools(self.tools)
        self.provider = ModelProvider(self.tool_llm)
        self.final_provider = ModelProvider(self.primary_llm)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.confirm_hook = ConfirmHook(self.pending_actions)

        self.context_builder = SystemContextBuilder(
            user_id=self.user_id,
            preference=self.preference,
            session_memory_store=self.session_memory_store,
            study_notes_store=self.study_notes_store,
            tools=self.tools,
            skill_loader=self.skill_loader,
        )
        self.runner = AgentRunner(
            provider=self.provider,
            tools_by_name=self.tools_by_name,
            tool_recovery=self.tool_recovery,
            hook=self.confirm_hook,
            final_provider=self.final_provider,
        )

    def _save_session(self) -> None:
        """把当前会话状态统一写回持久层。"""
        self.session.runtime.pending_actions = list(self.pending_actions)
        self.sessions.save_session(self.session)

    def _prune_expired_todos(self) -> bool:
        """清理当前 session 中已经过期且尚未完成的 todo。"""
        todos = self.session.runtime.todos
        if not todos:
            return False

        now = datetime.now()
        active_todos: list[dict[str, Any]] = []
        removed_count = 0

        # 只删除存在明确 end_at 且已经过期的未完成任务。
        for todo in todos:
            if self._is_expired_todo(todo, now):
                removed_count += 1
                continue
            active_todos.append(todo)

        if removed_count:
            self.session.runtime.todos = active_todos
            print(f"[Todo] 已清理 {removed_count} 条过期 todo")
            return True
        return False

    def _is_expired_todo(self, todo: dict[str, Any], now: datetime) -> bool:
        """判断单条 todo 是否已经过期。"""
        if str(todo.get("status", "todo")).strip().lower() == "done":
            return False

        time_scope = todo.get("time_scope") or {}
        if not isinstance(time_scope, dict):
            return False

        end_at_raw = time_scope.get("end_at")
        if not isinstance(end_at_raw, str) or not end_at_raw.strip():
            return False

        try:
            end_at = datetime.fromisoformat(end_at_raw)
        except ValueError:
            return False

        return end_at < now

    def _commit_turn_messages(self, turn_messages: list[dict[str, Any]], history_count: int) -> None:
        """把本轮新增消息提交回会话层。"""
        for message in turn_messages[history_count:]:
            if message.get("hidden"):
                continue
            extra = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content"}
            }
            self.session.add_message(
                message.get("role", ""),
                str(message.get("content", "")),
                **extra,
            )

    def get_chat_history(self) -> list[list[str]]:
        """把内部消息历史转成 Gradio 聊天框需要的成对结构。"""
        history: list[list[str]] = []
        pending_user_message: str | None = None

        for message in self.session.get_history():
            if message.get("hidden"):
                continue
            if message["role"] == "user":
                if pending_user_message is not None:
                    history.append([pending_user_message, ""])
                pending_user_message = message["content"]
            elif message["role"] == "assistant":
                if message.get("is_confirmation_result"):
                    continue
                if pending_user_message is not None:
                    history.append([pending_user_message, message["content"]])
                    pending_user_message = None
                else:
                    history.append(["", message["content"]])

        if pending_user_message is not None:
            history.append([pending_user_message, ""])
        return history

    def ask(
        self,
        user_message: str,
        on_content_delta: Callable[[str], None] | None = None,
    ) -> str:
        """执行一轮正常问答。"""
        self.session_stats["questions_asked"] += 1
        print(f"\n{'=' * 55}\n[USER] {user_message}\n{'=' * 55}")

        if self.is_first_user_turn:
            self.sessions.update_title(self.session, user_message)
            self.is_first_user_turn = False

        # 先保存用户输入，保证中途失败时消息不丢。
        # 如果上一条已经是相同内容的 user 消息（上轮失败残留），跳过重复写入。
        last_msg = self.session.messages[-1] if self.session.messages else None
        if not (last_msg and last_msg.get("role") == "user" and last_msg.get("content") == user_message):
            self.session.add_message("user", user_message)
        self.confirm_hook.start_turn()
        self._prune_expired_todos()

        # 轮次开始前先压缩一次热历史。
        self.session.messages = compress_conversation_window(
            self.session.messages,
            self.session_memory_store,
            self.user_id,
            self.session_id,
            self.fast_llm,
            recovery_manager=self.recovery,
        )
        self._save_session()

        system_prompt, session_memory_block = self.context_builder.build_system_prompt(
            session_id=self.session_id,
            user_message=user_message,
            loaded_document_names=self.loaded_document_names,
            todos=self.session.runtime.todos,
        )
        if session_memory_block:
            print(f"[Memory] 注入会话记忆: {session_memory_block[:80]}...")

        turn_messages = self.session.get_history()
        history_count = len(turn_messages)
        final_answer = self.runner.run(
            system_prompt=system_prompt,
            turn_messages=turn_messages,
            on_content_delta=on_content_delta,
        )

        self._commit_turn_messages(turn_messages, history_count)

        # 轮次结束后再压缩一次，为下一轮整理热历史。
        self.session.messages = compress_conversation_window(
            self.session.messages,
            self.session_memory_store,
            self.user_id,
            self.session_id,
            self.fast_llm,
            recovery_manager=self.recovery,
        )
        self._save_session()
        print(
            f"{'=' * 55}\n[最终回答]\n"
            f"{final_answer[:200]}{'...' if len(final_answer) > 200 else ''}\n{'=' * 55}\n"
        )
        return final_answer

    def get_current_pending_action(self) -> dict[str, Any] | None:
        """返回当前待确认队列头部动作。"""
        return self.pending_actions[0] if self.pending_actions else None

    def get_pending_action_summary(self) -> str:
        """把当前待确认动作转成前端可显示摘要。"""
        current_action = self.get_current_pending_action()
        if current_action is None:
            return "当前没有待确认操作。"

        remaining_count = max(len(self.pending_actions) - 1, 0)
        lines = [f"当前待确认：{current_action.get('summary', '未命名操作')}"]
        if remaining_count:
            lines.append(f"后续还有 {remaining_count} 项待确认。")
        return "\n".join(lines)

    def continue_pending_action(
        self,
        decision: str,
        feedback: str = "",
        on_content_delta: Callable[[str], None] | None = None,
    ) -> str:
        """处理当前待确认动作，并在后端静默继续一轮执行。"""
        current_action = self.get_current_pending_action()
        if current_action is None:
            return "当前没有待确认操作。"

        self.pending_actions.pop(0)
        self.confirm_hook.start_turn()
        system_prompt, _ = self.context_builder.build_system_prompt(
            session_id=self.session_id,
            user_message=feedback.strip() or current_action.get("summary", ""),
            loaded_document_names=self.loaded_document_names,
            todos=self.session.runtime.todos,
        )
        turn_messages = self.session.get_history()
        history_count = len(turn_messages)

        # 这段内部消息只给后端继续执行使用，不在前端渲染。
        turn_messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": current_action.get("tool_call_id", ""),
                        "name": current_action.get("tool_name", ""),
                        "args": dict(current_action.get("tool_args", {}) or {}),
                    }
                ],
                "hidden": True,
            }
        )
        turn_messages.append(
            {
                "role": "tool",
                "content": self._build_pending_tool_result(current_action, decision, feedback),
                "tool_call_id": current_action.get("tool_call_id", ""),
                "name": current_action.get("tool_name", ""),
                "hidden": True,
            }
        )
        if decision == "feedback" and feedback.strip():
            turn_messages.append(
                {
                    "role": "user",
                    "content": f"补充意见：{feedback.strip()}",
                    "hidden": True,
                }
            )

        final_answer = self.runner.run(
            system_prompt=system_prompt,
            turn_messages=turn_messages,
            on_content_delta=on_content_delta,
        )
        self._commit_turn_messages(turn_messages, history_count)
        self._save_session()
        return final_answer

    def _build_pending_tool_result(
        self,
        action: dict[str, Any],
        decision: str,
        feedback: str,
    ) -> str:
        """把前端确认结果转换成内部 tool result 文本。"""
        if decision == "approve":
            return self._execute_pending_action(action)
        if decision == "reject":
            return "用户拒绝执行该操作。"
        if feedback.strip():
            return "用户没有直接同意执行该操作，并给出了新的补充意见。"
        return "用户没有直接同意执行该操作。"

    def _execute_pending_action(self, action: dict[str, Any]) -> str:
        """真正执行已经同意的待确认工具。"""
        tool_name = str(action.get("tool_name", "")).strip()
        tool_args = action.get("tool_args", {}) or {}
        tool_impl = self.tools_by_name.get(tool_name)
        if tool_impl is None:
            return f"待确认工具 {tool_name} 尚未实现。"
        try:
            result = str(tool_impl.invoke(tool_args))
            print(f"[Confirm] 执行结果: {result}")
            return result
        except Exception as exc:
            print(f"[Confirm] 执行失败: {exc}")
            return f"待确认操作执行失败：{exc}"

    def load_document(self, pdf_path: str) -> dict[str, Any]:
        """供 UI 触发文档入库，并同步更新当前会话状态。"""
        load_result = load_document(
            pdf_path,
            self.pdf_store,
            self.user_id,
            fast_llm=self.fast_llm,
            lexical_index=self.lexical_index,
            recovery_manager=self.recovery,
        )
        if load_result["success"]:
            self.session_stats["docs_loaded"] += 1
            document_name = load_result["document"]
            if document_name not in self.loaded_document_names:
                self.loaded_document_names.append(document_name)
        return load_result

    def list_notes(self) -> list[dict[str, str]]:
        """列出当前用户的研究笔记。"""
        return list_study_notes(self.study_notes_store, self.user_id)

    def get_note(self, note_id: str) -> dict[str, str] | None:
        """按 note_id 读取单条研究笔记。"""
        return next((note for note in self.list_notes() if note["note_id"] == note_id), None)

    def save_note(self, content: str, title: str = "", note_id: str = "") -> str:
        """新增或更新研究笔记。"""
        cleaned_content = content.strip()
        if not cleaned_content:
            return "笔记内容不能为空。"

        if note_id.strip():
            updated = update_study_note(
                self.study_notes_store,
                self.user_id,
                note_id.strip(),
                cleaned_content,
                new_title=title,
            )
            return "笔记已更新。" if updated else "未找到对应的笔记。"

        created_note_id = save_study_note(
            self.study_notes_store,
            cleaned_content,
            self.user_id,
            source_session_id=self.session_id,
            title=title,
        )
        self.session_stats["notes_added"] += 1
        return f"笔记已保存，note_id={created_note_id}。"

    def delete_note(self, note_id: str) -> str:
        """删除指定研究笔记。"""
        deleted = delete_study_note(self.study_notes_store, self.user_id, note_id.strip())
        return "笔记已删除。" if deleted else "未找到对应的笔记。"
