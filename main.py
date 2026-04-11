# main.py
# 项目主入口，组装所有模块并启动 Gradio UI。
# MultiAgentApp 是核心协调类，持有所有模块的引用，
# 并提供 ask() 和 load_document() 两个对外接口。
# session_id 由外部传入，支持新建和恢复历史会话。

from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from config import get_llm, get_fast_llm, get_embeddings
from memory.store import init_qdrant, build_memory_context
from memory.compression import compress_window
from document.loader import load_document
from graph.builder import build_graph
from session.manager import create_session, update_session_title, get_db_path


class MultiAgentApp:
    def __init__(self, user_id: str = "default_user", session_id: str = None):
        self.user_id = user_id
        # 传入 session_id 则恢复历史会话，否则新建
        if session_id:
            self.session_id = session_id
            print(f"[Session] 恢复会话: {session_id}")
        else:
            self.session_id = create_session(user_id)

        self.llm = get_llm()
        self.fast_llm = get_fast_llm()
        embeddings = get_embeddings()

        self.qdrant_client, self.pdf_store, self.memory_store = init_qdrant(embeddings)

        self.stats = {
            "session_start": datetime.now(),
            "docs_loaded": 0,
            "questions_asked": 0,
            "notes_added": 0,
        }
        self.loaded_docs: list[str] = []  # 本次运行中加载的文档（供图内工具引用）
        self._first_message = True

        # SqliteSaver 持久化到本地文件，重启后会话历史不丢失
        conn = sqlite3.connect(get_db_path(), check_same_thread=False)
        self._db_conn = SqliteSaver(conn)
        self.app = build_graph(
            llm=self.llm,
            fast_llm=self.fast_llm,
            pdf_store=self.pdf_store,
            memory_store=self.memory_store,
            user_id=self.user_id,
            session_id=self.session_id,
            stats=self.stats,
            loaded_docs=self.loaded_docs,
            checkpointer=self._db_conn,
        )

    def get_chat_history(self) -> list:
        """从 checkpointer 读取当前 session 的消息历史，转换为 Gradio chatbot 格式。"""
        config = {"configurable": {"thread_id": self.session_id}}
        try:
            state = self.app.get_state(config)
            messages = state.values.get("messages", [])
        except Exception:
            return []

        history = []
        pending_user = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                pending_user = msg.content
            elif isinstance(msg, AIMessage) and msg.content and getattr(msg, "name", None):
                if pending_user is not None:
                    history.append([pending_user, msg.content])
                    pending_user = None
        return history

    def ask(self, question: str) -> str:
        self.stats["questions_asked"] += 1
        config = {"configurable": {"thread_id": self.session_id}}

        print(f"\n{'='*55}")
        print(f"[USER] {question}")
        print(f"{'='*55}")

        # 用第一条消息更新会话标题
        if self._first_message:
            update_session_title(self.session_id, question)
            self._first_message = False

        # 滑动窗口压缩（超出阈值时触发）
        compress_window(self.app, config, self.memory_store,
                        self.user_id, self.session_id, self.fast_llm)

        # 检索长期记忆注入上下文
        input_messages = []
        context = build_memory_context(self.memory_store, self.user_id, self.session_id, question)
        if context:
            print(f"[Memory] 注入长期记忆: {context[:80]}...")
            input_messages.append(SystemMessage(content=context))
        input_messages.append(HumanMessage(content=question))

        result = self.app.invoke(
            {
                "messages": input_messages,
                "plan": [],
                "plan_step": 0,
                "step_results": [],
            },
            config=config,
        )

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"{'='*55}")
                print(f"[最终回答] {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
                print(f"{'='*55}\n")
                return msg.content
        return "抱歉，未能生成回答。"

    def load_document(self, pdf_path: str) -> dict:
        result = load_document(pdf_path, self.pdf_store, self.user_id, fast_llm=self.fast_llm)
        if result["success"]:
            self.stats["docs_loaded"] += 1
            doc_name = result["document"]
            if doc_name not in self.loaded_docs:
                self.loaded_docs.append(doc_name)
        return result


if __name__ == "__main__":
    from ui.gradio_app import create_gradio_ui
    demo = create_gradio_ui(app_factory=MultiAgentApp)
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
