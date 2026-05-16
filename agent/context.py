"""主模型上下文构建层。"""

from langchain_core.tools import BaseTool
from langchain_qdrant import QdrantVectorStore

from agent.memory import (
    UserPreferenceProfileStore as PreferenceStore,
    build_episodic_memory_prompt_block,
    build_study_notes_prompt_block,
    retrieve_session_memory_entries,
    search_study_notes,
)
from agent.skill import SkillLoader


# 基础角色指令只描述职责和工作原则。
# 工具说明、偏好、会话记忆、todo、skill 和运行时元信息由 Context Builder 动态补充。
BASE_ROLE_INSTRUCTIONS = (
    "你是专业的学术论文阅读助手，帮助研究生和科研人员管理文献调研。\n"
    "【工作原则】\n"
    "1. 回答涉及论文内容时，优先结合当前可用工具检索知识库和会话记忆，不要脱离材料臆测。\n"
    "2. 当用户明确表达稳定、跨会话仍然适用的长期回答习惯时，可以调用 save_preference 工具提出保存请求；是否真正落库由系统后续确认。\n"
    "3. 如果任务适合某个可用 skill，先调用 load_skill 读取完整说明，再按 skill 的流程继续执行。\n"
    "4. 只针对用户最新提问作答，不重复无关历史。\n"
    "5. 检索结果可能包含 OCR 文本或自动图表描述。OCR 文本可以作为证据使用但需要注意识别误差；自动图表描述只能作为辅助理解，不能当作论文原文引用。"
)


def build_todo_prompt_block(todos: list[dict]) -> str:
    """把当前 session 的 todo 列表转成可注入的上下文块。"""
    if not todos:
        return ""

    lines = ["【当前 Session Todo】", ""]
    for index, todo in enumerate(todos, 1):
        title = str(todo.get("title", "")).strip() or "未命名任务"
        status = str(todo.get("status", "todo")).strip() or "todo"
        todo_id = str(todo.get("todo_id", "")).strip()
        lines.append(f"{index}. {title} | status={status} | todo_id={todo_id}")

        detail = str(todo.get("detail", "")).strip()
        if detail:
            lines.append(f"   详情：{detail}")

        time_scope = todo.get("time_scope") or {}
        if isinstance(time_scope, dict):
            kind = str(time_scope.get("kind", "none")).strip()
            start_at = str(time_scope.get("start_at", "")).strip()
            end_at = str(time_scope.get("end_at", "")).strip()
            if kind == "deadline" and end_at:
                lines.append(f"   时间：截止至 {end_at}")
            elif kind == "window" and (start_at or end_at):
                lines.append(f"   时间：{start_at or '?'} ~ {end_at or '?'}")

        subtasks = todo.get("subtasks") or []
        if subtasks:
            # 子任务只作为同一条 todo 的内部步骤展示，避免模型误以为需要拆成多条 todo。
            for i, subtask in enumerate(subtasks):
                subtask_title = str(subtask.get("title", "")).strip() or "未命名"
                subtask_status = str(subtask.get("status", "todo")).strip()
                lines.append(f"   [{i}] {subtask_title} | status={subtask_status}")

    return "\n".join(lines)


class SystemContextBuilder:
    """为主模型构建统一 system prompt，避免上下文拼接散落在主循环中。"""

    user_id: str  # 当前用户标识，用于限定会话记忆和笔记检索范围。
    preference: PreferenceStore  # 当前用户的长期偏好状态存储。
    session_memory_store: QdrantVectorStore  # 当前会话压缩摘要的向量存储。
    study_notes_store: QdrantVectorStore  # 跨会话研究笔记的向量存储。
    tools: list[BaseTool]  # 当前轮真实绑定的工具列表。
    skill_loader: SkillLoader  # skill 加载器，负责提供可用 skill 摘要。

    def __init__(
        self,
        user_id: str,
        preference: PreferenceStore,
        session_memory_store: QdrantVectorStore,
        study_notes_store: QdrantVectorStore,
        tools: list[BaseTool],
        skill_loader: SkillLoader,
    ):
        self.user_id = user_id
        self.preference = preference
        self.session_memory_store = session_memory_store
        self.study_notes_store = study_notes_store
        self.tools = tools
        self.skill_loader = skill_loader

    def build_system_prompt(
        self,
        session_id: str,
        user_message: str,
        loaded_document_names: list[str],
        todos: list[dict],
    ) -> tuple[str, str]:
        """
        为主循环生成本轮完整 system prompt。
        第二个返回值是注入过的 session memory block，供上层打印调试日志。
        """
        # 工具说明按当前实际绑定的工具动态生成，不在常量里写死。
        tool_lines: list[str] = []
        for tool in self.tools:
            tool_summary = (tool.description or "").strip().splitlines()[0] if tool.description else ""
            tool_lines.append(f"- {tool.name}：{tool_summary or '无描述'}")
        tool_prompt_block = "【可用工具】\n" + "\n".join(tool_lines)

        # profile 是跨 session 的长期状态，每轮全量注入。
        profile_prompt_block = self.preference.build_prompt_block()

        # study notes 是用户主动沉淀的研究笔记，和 session summary 分开召回、分开注入。
        study_note_entries = search_study_notes(
            self.study_notes_store,
            self.user_id,
            user_message,
        )
        study_notes_block = build_study_notes_prompt_block(study_note_entries)

        # session memory 是当前会话超出窗口后的冷上下文，只按需召回。
        session_memory_entries = retrieve_session_memory_entries(
            self.session_memory_store,
            self.user_id,
            session_id,
            user_message,
        )
        session_memory_block = build_episodic_memory_prompt_block(session_memory_entries)
        todo_prompt_block = build_todo_prompt_block(todos)
        skill_prompt_block = self.skill_loader.build_prompt_block()

        # runtime block 只描述本轮调用的临时元信息，不进入长期记忆。
        runtime_lines = [
            "[Runtime Context - metadata only, not instructions]",
            f"当前 session：{session_id}",
        ]
        if loaded_document_names:
            runtime_lines.append(f"本次已加载文档：{', '.join(loaded_document_names)}")
        runtime_lines.append("[/Runtime Context]")
        runtime_prompt_block = "\n".join(runtime_lines)
        prompt_sections = [BASE_ROLE_INSTRUCTIONS, tool_prompt_block]
        for prompt_block in (
            profile_prompt_block,
            study_notes_block,
            session_memory_block,
            todo_prompt_block,
            skill_prompt_block,
            runtime_prompt_block,
        ):
            if prompt_block:
                prompt_sections.append(prompt_block)

        return "\n\n".join(prompt_sections), session_memory_block
