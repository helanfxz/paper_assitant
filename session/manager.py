# session/manager.py
# 负责 session 的生命周期管理：创建、加载、列举、持久化元数据。
# session 元数据（id、标题、创建时间）存储在 sessions.json 中，
# 消息历史通过 SqliteSaver 持久化到 sessions.db，两者通过 session_id 关联。
# 每个 session 完全独立，不同 session 之间不共享任何消息或记忆。

import json
import os
from datetime import datetime

SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "sessions.json")
SESSIONS_DB = os.path.join(os.path.dirname(__file__), "..", "sessions.db")


def _load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_sessions(sessions: dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def create_session(user_id: str, title: str = "") -> str:
    """创建新 session，返回 session_id。"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sessions = _load_sessions()
    sessions[session_id] = {
        "user_id": user_id,
        "title": title or "新对话",
        "created_at": datetime.now().isoformat(),
    }
    _save_sessions(sessions)
    print(f"[Session] 新建会话: {session_id}")
    return session_id


def update_session_title(session_id: str, first_message: str):
    """用第一条用户消息的前 20 字作为会话标题。"""
    sessions = _load_sessions()
    if session_id in sessions and sessions[session_id]["title"] == "新对话":
        sessions[session_id]["title"] = first_message[:20]
        _save_sessions(sessions)


def list_sessions(user_id: str) -> list:
    """返回该用户的所有 session，按创建时间倒序。"""
    sessions = _load_sessions()
    user_sessions = [
        {"session_id": sid, **info}
        for sid, info in sessions.items()
        if info.get("user_id") == user_id
    ]
    return sorted(user_sessions, key=lambda x: x["created_at"], reverse=True)


def get_db_path() -> str:
    return os.path.abspath(SESSIONS_DB)
