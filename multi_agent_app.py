# multi_agent_app.py
# 已被新模块化结构取代，入口改为 main.py。
# 保留此文件仅供参考，不再维护。

from dotenv import load_dotenv
load_dotenv()

import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore

import pymupdf4llm


# ==========================================
# Multi-Agent 共享状态
# ==========================================
class SupervisorState(TypedDict):
    """
    所有 Agent 共享同一份消息列表。
    Supervisor 通过 next 字段控制路由。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    next: str  # "ResearchAgent" | "MemoryAgent" | "GeneralAgent" | "FINISH"


# ==========================================
# 核心架构：Multi-Agent 实现
# ==========================================
class MultiAgentPDFLearningAgent:
    """
    Supervisor + 专家 Agent 架构：
      SupervisorAgent  —— 负责意图识别与任务路由
      ResearchAgent    —— 负责 PDF 文档检索与学术问答
      MemoryAgent      —— 负责历史记忆查询与笔记管理
      GeneralAgent     —— 负责日常闲聊与统计查询
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.embeddings = OpenAIEmbeddings(
            model="Embedding-3",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_URL"),
            check_embedding_ctx_length=False
        )
        self.llm = ChatOpenAI(
            model="glm-5",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_URL"),
        )
        self.fast_llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_URL"),
            temperature=1.0,
            max_tokens=65536,
        )

        qdrant_url = os.getenv("QDRANT_URL", ":memory:")
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
        self._init_qdrant_collections()

        self.pdf_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name="pdf_knowledge", embedding=self.embeddings
        )
        self.memory_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name="user_semantic_memory", embedding=self.embeddings
        )

        self.stats = {
            "session_start": datetime.now(),
            "docs_loaded": 0,
            "questions_asked": 0,
            "notes_added": 0
        }
        self.current_document = None
        self._last_retrieved_docs = ""

        self.checkpointer = MemorySaver()
        self.app = self._build_multi_agent_graph()

    def _init_qdrant_collections(self):
        collections = [col.name for col in self.qdrant_client.get_collections().collections]
        for col_name in ["pdf_knowledge", "user_semantic_memory"]:
            if col_name not in collections:
                self.qdrant_client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
                )

    # ==========================================================
    # 工具定义：按职责分组，分配给不同的专家 Agent
    # ==========================================================

    def _make_research_tools(self):
        """ResearchAgent 专用工具：PDF 检索"""
        agent_self = self

        @tool
        def search_pdf(query: str) -> str:
            """在已加载的 PDF 文档中检索与问题相关的内容。"""
            print(f"[ResearchAgent][TOOL] search_pdf: {query}")
            mqe_prompt = f"为了在学术文档中全面检索以下问题，请生成3个不同表达或侧重点的相似搜索词。不要加序号，每行一个。\n原始问题：{query}"
            extended = agent_self.fast_llm.invoke(mqe_prompt).content.split("\n")
            search_queries = [query] + [q.strip() for q in extended if q.strip()]

            all_child_docs = []
            for q in search_queries:
                all_child_docs.extend(agent_self.pdf_store.similarity_search(q, k=3))

            parent_map = {}
            for doc in all_child_docs:
                pid = doc.metadata.get("parent_id")
                if pid and pid not in parent_map:
                    parent_map[pid] = doc.metadata.get("parent_text", doc.page_content)

            unique_parents = list(parent_map.values())
            if not unique_parents:
                agent_self._last_retrieved_docs = ""
                return "未能检索到相关文档内容。"

            rerank_prompt = f"以下是从文档中粗筛出的几个片段。请挑选出与问题【{query}】最相关的片段并拼接，剔除无关片段。\n\n片段：\n{unique_parents[:8]}"
            result = agent_self.fast_llm.invoke(rerank_prompt).content
            agent_self._last_retrieved_docs = result
            return result

        return [search_pdf]

    def _make_memory_tools(self):
        """MemoryAgent 专用工具：记忆查询 + 笔记管理"""
        agent_self = self

        @tool
        def recall_memory(query: str) -> str:
            """从用户的历史笔记和对话记录中检索相关记忆。"""
            print(f"[MemoryAgent][TOOL] recall_memory: {query}")
            filter_kwargs = {"filter": {"must": [{"key": "user_id", "match": {"value": agent_self.user_id}}]}}
            try:
                docs = agent_self.memory_store.similarity_search(query, k=4, **filter_kwargs)
            except Exception:
                docs = agent_self.memory_store.similarity_search(query, k=4)
            return "\n\n".join([f"[{d.metadata.get('type', 'note')}]: {d.page_content}" for d in docs])

        @tool
        def add_note(content: str) -> str:
            """将用户提供的内容保存为学习笔记到记忆库。"""
            print(f"[MemoryAgent][TOOL] add_note: {content[:50]}")
            doc = Document(
                page_content=content,
                metadata={"user_id": agent_self.user_id, "type": "note", "concept": "general"}
            )
            agent_self.memory_store.add_documents([doc])
            agent_self.stats["notes_added"] += 1
            return f"笔记已保存：{content[:50]}..."

        return [recall_memory, add_note]

    def _make_general_tools(self):
        """GeneralAgent 专用工具：统计查询"""
        agent_self = self

        @tool
        def get_stats() -> str:
            """获取当前会话的学习统计信息。"""
            s = agent_self.stats
            duration = (datetime.now() - s["session_start"]).seconds
            return (f"会话时长: {duration}秒 | 加载文档: {s['docs_loaded']} | "
                    f"提问次数: {s['questions_asked']} | 笔记数量: {s['notes_added']}")

        return [get_stats]

    # ==========================================================
    # 专家 Agent 构建：每个 Agent 是独立的 ReAct 子图
    # ==========================================================

    def _build_sub_agent(self, tools: list, system_prompt: str, name: str = "Agent"):
        """
        通用子图工厂：构建一个标准 ReAct 子图。
        结构：START → agent ⇄ tools → END
        """
        llm_with_tools = self.llm.bind_tools(tools)
        tool_node = ToolNode(tools)
        sys_msg = SystemMessage(content=system_prompt)

        def agent_node(state: MessagesState):
            response = llm_with_tools.invoke([sys_msg] + state["messages"])
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    print(f"  [{name}] 决策: 调用工具 {tc['name']} | 参数: {tc['args']}")
            else:
                print(f"  [{name}] 决策: 生成最终回答")
            return {"messages": [response]}

        def should_continue(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                print(f"  [{name}] → 执行工具节点")
                return "tools"
            print(f"  [{name}] → 回答完成，返回主图")
            return END

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")
        return graph.compile()

    # ==========================================================
    # Supervisor：LLM 路由节点
    # ==========================================================

    def _build_supervisor_node(self):
        """
        Supervisor 节点：读取对话历史，决定下一个执行的 Agent。
        路由逻辑完全由 LLM 驱动，不依赖关键词匹配。
        """
        def supervisor_node(state: SupervisorState) -> dict:
            # 格式化对话历史供 Supervisor 参考
            history_lines = []
            for m in state["messages"]:
                if isinstance(m, HumanMessage):
                    history_lines.append(f"用户: {m.content}")
                elif isinstance(m, AIMessage):
                    name = getattr(m, "name", None) or "AI"
                    history_lines.append(f"{name}: {m.content}")
            history = "\n".join(history_lines)
            print(f"历史信息:{history}")
            prompt = (
                f"你是系统的主控路由节点。你的唯一任务是决定下一步的任务走向。\n\n"
                f"【对话历史】：\n{history}\n\n"
                f"【严格判断逻辑】：\n"
                f"请重点关注对话历史的【最后一条消息】！\n"
                f"🚨 规则1：如果最后一条消息是 'ResearchAgent'、'MemoryAgent' 或 'GeneralAgent' 发出的，并且它已经回答了用户的问题，你必须立刻输出：FINISH\n"
                f"规则2：只有当用户的最新问题【还没有被任何人回答】时，你才从以下专家中选择一个派发：\n"
                f"- ResearchAgent：处理 PDF 检索和学术问题\n"
                f"- MemoryAgent：处理历史记忆和记笔记\n"
                f"- GeneralAgent：处理闲聊和查询统计\n\n"
                f"【输出要求】：\n"
                f"仅输出一个词（FINISH, ResearchAgent, MemoryAgent, GeneralAgent），绝不能包含其他任何字符！\n"
                f"输出："
            )

            decision = self.fast_llm.invoke(prompt).content.strip()
            print(f"[Supervisor] 路由决策：{decision}")

            for agent in ["ResearchAgent", "MemoryAgent", "GeneralAgent", "FINISH"]:
                if agent in decision:
                    return {"next": agent}
            return {"next": "FINISH"}

        return supervisor_node

    # ==========================================================
    # 主图构建：Supervisor + 三个专家 Agent
    # ==========================================================

    def _build_multi_agent_graph(self):
        # 构建三个专家子图
        research_agent = self._build_sub_agent(
            tools=self._make_research_tools(),
            system_prompt=(
                "你是专业的学术文档检索专家。\n"
                "【🚨 核心工作纪律】：\n"
                "传给你的消息列表中包含了过去的对话历史。这些历史【仅仅是为了让你理解上下文中的代词（如：它、这个）】。\n"
                "你**必须且只能**针对用户的【最后一条最新提问】进行回答和工具调用！\n"
                "绝对禁止去回答或检索历史记录中已经出现过的问题！"
            ),
            name="ResearchAgent"
        )
        memory_agent = self._build_sub_agent(
            tools=self._make_memory_tools(),
            system_prompt=(
                "你是用户的个人记忆管理专家。"
                "使用 recall_memory 查询历史笔记和对话，"
                "使用 add_note 保存用户的新笔记。"
            ),
            name="MemoryAgent"
        )
        general_agent = self._build_sub_agent(
            tools=self._make_general_tools(),
            system_prompt=(
                "你是友好的通用助手。"
                "可以使用 get_stats 查询学习统计，"
                "也可以直接回答日常问题，无需工具。"
            ),
            name="GeneralAgent"
        )

        # 将子图包装为主图的节点
        # 每个节点：调用子图 → 取最终回答 → 以 Agent 名义追加到共享消息列表
        def research_node(state: SupervisorState) -> dict:
            print("[Supervisor] → ResearchAgent")
            result = research_agent.invoke({"messages": state["messages"]})
            answer = result["messages"][-1].content
            print(f"[ResearchAgent] 回答: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            return {"messages": [AIMessage(content=answer, name="ResearchAgent")]}

        def memory_node(state: SupervisorState) -> dict:
            print("[Supervisor] → MemoryAgent")
            result = memory_agent.invoke({"messages": state["messages"]})
            answer = result["messages"][-1].content
            print(f"[MemoryAgent] 回答: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            return {"messages": [AIMessage(content=answer, name="MemoryAgent")]}

        def general_node(state: SupervisorState) -> dict:
            print("[Supervisor] → GeneralAgent")
            result = general_agent.invoke({"messages": state["messages"]})
            answer = result["messages"][-1].content
            print(f"[GeneralAgent] 回答: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            return {"messages": [AIMessage(content=answer, name="GeneralAgent")]}

        # 构建主图
        workflow = StateGraph(SupervisorState)

        # 节点注册
        workflow.add_node("supervisor", self._build_supervisor_node())
        workflow.add_node("ResearchAgent", research_node)
        workflow.add_node("MemoryAgent", memory_node)
        workflow.add_node("GeneralAgent", general_node)

        # 入口：用户问题先交给 Supervisor
        workflow.add_edge(START, "supervisor")

        # Supervisor 条件路由
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "ResearchAgent": "ResearchAgent",
                "MemoryAgent":   "MemoryAgent",
                "GeneralAgent":  "GeneralAgent",
                "FINISH":        END,
            }
        )

        # 每个专家完成后回到 Supervisor，由 Supervisor 决定是否继续或结束
        workflow.add_edge("ResearchAgent", "supervisor")
        workflow.add_edge("MemoryAgent",   "supervisor")
        workflow.add_edge("GeneralAgent",  "supervisor")

        return workflow.compile(checkpointer=self.checkpointer)

    # ==========================================================
    # 文档入库（与 app.py 相同）
    # ==========================================================
    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}
        start_time = time.time()
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            parent_docs = parent_splitter.create_documents([md_text])
            child_docs = []
            for p_doc in parent_docs:
                p_id = str(uuid.uuid4())
                p_text = p_doc.page_content
                for c in child_splitter.create_documents([p_text]):
                    c.metadata = {
                        "parent_id": p_id, "parent_text": p_text,
                        "source": os.path.basename(pdf_path), "user_id": self.user_id
                    }
                    child_docs.append(c)
            self.pdf_store.add_documents(child_docs)
            self.current_document = os.path.basename(pdf_path)
            self.stats["docs_loaded"] += 1
            process_time = time.time() - start_time
            return {"success": True,
                    "message": f"解析成功 (耗时: {process_time:.1f}s)，入库 {len(child_docs)} 个子块",
                    "document": self.current_document}
        except Exception as e:
            return {"success": False, "message": str(e)}

    # ==========================================================
    # 对外接口
    # ==========================================================
    def ask(self, question: str) -> str:
        self.stats["questions_asked"] += 1
        print(f"\n{'='*55}")
        print(f"[USER] {question}")
        print(f"{'='*55}")
        config = {"configurable": {"thread_id": self.session_id}}
        result = self.app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        # 取最后一条有内容的 AI 消息作为最终回答
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"{'='*55}")
                print(f"[最终回答] {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
                print(f"{'='*55}\n")
                return msg.content
        return "抱歉，未能生成回答。"

    def add_note(self, content: str, concept: Optional[str] = None):
        doc = Document(
            page_content=content,
            metadata={"user_id": self.user_id, "type": "note", "concept": concept or "general"}
        )
        self.memory_store.add_documents([doc])


# ==========================================
# 前端交互：Gradio UI 层
# ==========================================
import gradio as gr


def create_gradio_ui():
    assistant_state = {"assistant": None}

    def init_assistant(user_id: str) -> str:
        if not user_id:
            user_id = "pdf_user1"
        assistant_state["assistant"] = MultiAgentPDFLearningAgent(user_id=user_id)
        return f"✅ Multi-Agent 助手已初始化 (用户: {user_id} | 引擎: Supervisor+ReAct)"

    def load_pdf(pdf_file) -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        if pdf_file is None:
            return "❌ 请上传PDF文件"
        result = assistant_state["assistant"].load_document(pdf_file.name)
        if result["success"]:
            return f"✅ {result['message']}\n📄 文档: {result['document']}"
        return f"❌ {result['message']}"

    def chat(message: str, history: List) -> Tuple[str, List]:
        if assistant_state["assistant"] is None:
            return "", history + [[message, "❌ 请先初始化助手"]]
        if not message.strip():
            return "", history
        response = assistant_state["assistant"].ask(message)
        history.append([message, response])
        return "", history

    def add_note_ui(note_content: str, concept: str) -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        if not note_content.strip():
            return "❌ 笔记内容不能为空"
        assistant_state["assistant"].add_note(note_content, concept or None)
        return f"✅ 笔记已保存: {note_content[:50]}..."

    with gr.Blocks(title="Multi-Agent 文档问答助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 Multi-Agent 文档问答助手 (Supervisor + 专家 Agent 架构)")

        with gr.Tab("🏠 开始使用"):
            with gr.Row():
                user_id_input = gr.Textbox(label="用户ID", value="pdf_user1", scale=3)
                init_btn = gr.Button("初始化助手", variant="primary", scale=1)
            init_output = gr.Textbox(label="系统状态", interactive=False)
            init_btn.click(init_assistant, inputs=[user_id_input], outputs=[init_output])

            gr.Markdown("### 📄 加载PDF文档")
            pdf_upload = gr.File(label="上传PDF文件", file_types=[".pdf"], type="filepath")
            load_btn = gr.Button("加载文档", variant="primary")
            load_output = gr.Textbox(label="加载状态", interactive=False)
            load_btn.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])

        with gr.Tab("💬 智能问答"):
            chatbot = gr.Chatbot(label="对话历史", height=400)
            with gr.Row():
                msg_input = gr.Textbox(label="输入问题", scale=4)
                send_btn = gr.Button("发送", variant="primary", scale=1)
            msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

        with gr.Tab("📝 学习笔记"):
            note_content = gr.Textbox(label="笔记内容", lines=3)
            concept_input = gr.Textbox(label="相关概念（可选）")
            note_btn = gr.Button("保存笔记", variant="primary")
            note_output = gr.Textbox(label="保存状态", interactive=False)
            note_btn.click(add_note_ui, inputs=[note_content, concept_input], outputs=[note_output])

    return demo


def main():
    demo = create_gradio_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)


if __name__ == "__main__":
    main()
