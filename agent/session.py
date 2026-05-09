# session.py
# 当前项目的会话状态层。
# 这一层负责两件事：
# 1. 持久化“这条对话已经稳定保存了哪些消息”
# 2. 持久化“这条对话当前还有哪些待处理运行时状态”

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent  # 当前项目的数据根目录。
SESSIONS_FILE = BASE_DIR / "sessions.json"  # 会话元数据和运行时状态的索引文件。
SESSIONS_DB = BASE_DIR / "sessions.db"  # 会话消息历史的 SQLite 数据库文件。
DEFAULT_SESSION_TITLE = "新对话"  # 新建会话时使用的默认标题。


@dataclass
class Runtime:
    """当前对话正在进行中的运行时状态。"""

    pending_actions: list[dict[str, Any]] = field(default_factory=list)  # 待用户确认的动作队列。
    todos: list[dict[str, Any]] = field(default_factory=list)  # 当前 session 内持久化保存的 todo 列表。

    def to_dict(self) -> dict[str, Any]:
        """把运行时状态转成可落盘的普通字典。"""
        return {
            "pending_actions": self.pending_actions,
            "todos": self.todos,
        }

    @classmethod
    def from_dict(cls, raw_data: dict[str, Any] | None) -> "Runtime":
        """从落盘字典恢复运行时状态。"""
        raw_data = raw_data or {}
        pending_actions = raw_data.get("pending_actions") or []
        todos = raw_data.get("todos") or []
        legacy_pending = raw_data.get("pending_confirmation")
        # 兼容旧版单个 pending_confirmation，避免升级后旧会话直接丢状态。
        if legacy_pending and not pending_actions:
            pending_actions = [
                {
                    "approval_id": "LEGACY-PREF-1",
                    "kind": "preference",
                    "tool_name": "save_preference",
                    "tool_args": legacy_pending,
                    "summary": "历史遗留的偏好确认操作",
                }
            ]
        return cls(
            pending_actions=list(pending_actions),
            todos=list(todos),
        )


@dataclass
class Session:
    """一条对话的统一承载对象。"""

    session_id: str  # 这条对话的唯一标识。
    user_id: str  # 这条对话所属的用户标识。
    title: str = DEFAULT_SESSION_TITLE  # 前端会话列表中显示的标题。
    messages: list[dict[str, Any]] = field(default_factory=list)  # 当前会话已经稳定保存的消息历史。
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())  # 会话创建时间。
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())  # 会话最后一次保存时间。
    metadata: dict[str, Any] = field(default_factory=dict)  # 预留给后续扩展的附加状态字典。
    runtime: Runtime = field(default_factory=Runtime)  # 当前会话的运行时状态对象。

    def is_new(self) -> bool:
        """判断这条对话是否还没有进入真实问答阶段。"""
        return self.title == DEFAULT_SESSION_TITLE and not self.messages

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        """向当前会话追加一条稳定消息，并同步刷新更新时间。"""
        message = {"role": role, "content": content}
        message.update(extra)
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

    def get_history(self) -> list[dict[str, Any]]:
        """
        返回当前会话里稳定保存的消息历史。
        这里先不做 nanobot 那种边界修正；这一轮只先建立统一 Session 对象。
        """
        return [
            dict(message)
            for message in self.messages
            if not message.get("hidden")
        ]


class SessionManager:
    """统一管理会话元数据、消息历史和运行时状态。"""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or BASE_DIR
        self.meta_path = self.base_dir / "sessions.json"
        self.db_path = self.base_dir / "sessions.db"

    def _load_meta_map(self) -> dict[str, dict[str, Any]]:
        """读取所有会话的元数据索引。"""
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def _save_meta_map(self, meta_map: dict[str, dict[str, Any]]) -> None:
        """把会话元数据索引整体写回磁盘。"""
        self.meta_path.write_text(
            json.dumps(meta_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_conn(self) -> sqlite3.Connection:
        """获取消息历史数据库连接，并确保消息表存在。"""
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                ts         TEXT,
                payload    TEXT
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "payload" not in columns:
            connection.execute("ALTER TABLE messages ADD COLUMN payload TEXT")
        connection.commit()
        return connection

    def _load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """从 SQLite 读取一条对话的消息历史。"""
        connection = self._get_conn()
        rows = connection.execute(
            "SELECT role, content, payload FROM messages WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall()
        connection.close()

        messages: list[dict[str, Any]] = []
        # 逐条恢复落盘消息。优先恢复完整 payload，
        # 兼容旧数据时再退回只含 role/content 的历史格式。
        for role, content, payload in rows:
            if payload:
                try:
                    restored_message = json.loads(payload)
                    if isinstance(restored_message, dict):
                        messages.append(restored_message)
                        continue
                except json.JSONDecodeError:
                    pass
            messages.append({"role": role, "content": content})
        return [message for message in messages if not message.get("hidden")]

    def _save_messages(self, session: Session) -> None:
        """
        覆盖写入一条对话的消息历史。
        当前项目仍然按“整轮结束后整体覆盖”的方式保存消息，先保证行为简单清楚。
        """
        connection = self._get_conn()
        connection.execute("DELETE FROM messages WHERE session_id = ?", (session.session_id,))
        now = datetime.now().isoformat()

        # 逐条写入当前会话的稳定消息历史，目的是让重新打开旧会话时能恢复到当前进度。
        for message in session.messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = ""
            connection.execute(
                "INSERT INTO messages (session_id, role, content, ts, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    session.session_id,
                    role,
                    content,
                    now,
                    json.dumps(message, ensure_ascii=False),
                ),
            )

        connection.commit()
        connection.close()

    def create_session(self, user_id: str) -> Session:
        """新建一条空会话，并立刻写入元数据索引。"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = Session(session_id=session_id, user_id=user_id)
        self.save_session(session)
        print(f"[Session] 新建会话: {session_id}")
        return session

    def load_session(self, session_id: str) -> Session | None:
        """按 session_id 读取一条完整会话。"""
        meta_map = self._load_meta_map()
        raw_meta = meta_map.get(session_id)
        if raw_meta is None:
            return None

        return Session(
            session_id=session_id,
            user_id=raw_meta.get("user_id", "default_user"),
            title=raw_meta.get("title", DEFAULT_SESSION_TITLE),
            messages=self._load_messages(session_id),
            created_at=raw_meta.get("created_at", datetime.now().isoformat()),
            updated_at=raw_meta.get("updated_at", datetime.now().isoformat()),
            metadata=raw_meta.get("metadata", {}),
            runtime=Runtime.from_dict(raw_meta.get("runtime")),
        )

    def get_or_create(self, session_id: str | None, user_id: str) -> Session:
        """
        统一获取会话入口。
        如果传入了已有 session_id，就恢复旧会话；否则创建新会话。
        """
        if session_id:
            loaded_session = self.load_session(session_id)
            if loaded_session is not None:
                return loaded_session
        return self.create_session(user_id)

    def save_session(self, session: Session) -> None:
        """
        保存一条完整会话。
        会话元数据和运行时状态写到 sessions.json，消息历史写到 sessions.db。
        """
        session.updated_at = datetime.now().isoformat()
        meta_map = self._load_meta_map()
        meta_map[session.session_id] = {
            "user_id": session.user_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "metadata": session.metadata,
            "runtime": session.runtime.to_dict(),
        }
        self._save_meta_map(meta_map)
        self._save_messages(session)

    def update_title(self, session: Session, first_message: str) -> None:
        """只在标题还是默认值时，按首条消息生成会话标题。"""
        if session.title == DEFAULT_SESSION_TITLE:
            session.title = first_message[:20]
            self.save_session(session)

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """列出某个用户的所有会话，供前端会话选择器使用。"""
        meta_map = self._load_meta_map()
        sessions = [
            {"session_id": session_id, **info}
            for session_id, info in meta_map.items()
            if info.get("user_id") == user_id
        ]
        return sorted(sessions, key=lambda item: item.get("created_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """删除一条会话的元数据和消息历史。"""
        meta_map = self._load_meta_map()
        existed = session_id in meta_map
        if existed:
            del meta_map[session_id]
            self._save_meta_map(meta_map)

        connection = self._get_conn()
        connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        connection.commit()
        connection.close()
        return existed


_SESSIONS = SessionManager()


def get_sessions() -> SessionManager:
    """提供全局会话管理器，兼容当前项目的轻量单例用法。"""
    return _SESSIONS


def create_session(user_id: str) -> str:
    """兼容旧接口：创建会话并返回 session_id。"""
    return _SESSIONS.create_session(user_id).session_id


def update_session_title(session_id: str, first_message: str) -> None:
    """兼容旧接口：按首条消息更新标题。"""
    session = _SESSIONS.load_session(session_id)
    if session is not None:
        _SESSIONS.update_title(session, first_message)


def list_sessions(user_id: str) -> list[dict[str, Any]]:
    """兼容旧接口：列出用户会话。"""
    return _SESSIONS.list_sessions(user_id)


def delete_session(session_id: str) -> bool:
    """兼容旧接口：删除会话。"""
    return _SESSIONS.delete_session(session_id)


def save_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    """兼容旧接口：只覆盖写消息历史，不主动改动其它会话状态。"""
    session = _SESSIONS.load_session(session_id)
    if session is None:
        return
    session.messages = list(messages)
    _SESSIONS.save_session(session)


def load_messages(session_id: str) -> list[dict[str, Any]]:
    """兼容旧接口：读取消息历史。"""
    session = _SESSIONS.load_session(session_id)
    return session.get_history() if session is not None else []
