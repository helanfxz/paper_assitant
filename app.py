
from dotenv import load_dotenv
# 加载环境变量 (需要 ZHIPU_API_KEY, QDRANT_URL, QDRANT_API_KEY)
load_dotenv()

import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# 引入 LangChain & LangGraph 核心库
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState, START, END  # [优化 #1] 手动构建 ReAct 图
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 引入 Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore

# 引入专为大模型优化的 PDF 解析库
import pymupdf4llm


# ==========================================
# 核心架构：ReAct Agent 实现
# ==========================================
class IndustrialPDFLearningAgent:
    """基于 LangGraph + Qdrant 的企业级多重优化文档问答助手"""

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
        self.fast_llm=ChatOpenAI(
            model="glm-4-flash",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_URL"),
            temperature=1.0,
            max_tokens=65536,
        )
        """
        self.rerank_llm=ChatOpenAI(
            model="rerank",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_URL"),
        )
        """
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
        self._last_retrieved_docs = ""  # [优化 #1] 供幻觉检测使用

        self.checkpointer = MemorySaver()
        self.app = self._build_agent()  # [优化 #1] ReAct Agent

    def _init_qdrant_collections(self):
        collections = [col.name for col in self.qdrant_client.get_collections().collections]
        for col_name in ["pdf_knowledge", "user_semantic_memory"]:
            if col_name not in collections:
                self.qdrant_client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(size=2048, distance=Distance.COSINE),  # 根据你的模型调整维度
                )

    # ---------------------------------------------------------
    # [优化 #1] 工具定义：将核心操作封装为 LangChain Tool
    # ---------------------------------------------------------
    def _build_tools(self):
        agent_self = self  # 闭包捕获 self

        @tool
        def search_pdf(query: str) -> str:
            """在已加载的 PDF 文档中检索与问题相关的内容。当用户询问学术知识、文档内容时使用。"""
            print(f"[TOOL] search_pdf: {query}")
            # 多查询扩展 (MQE)
            mqe_prompt = f"为了在学术文档中全面检索以下问题，请生成3个不同表达或侧重点的相似搜索词。不要加序号，每行一个。\n原始问题：{query}"
            extended = agent_self.fast_llm.invoke(mqe_prompt).content.split("\n")
            search_queries = [query] + [q.strip() for q in extended if q.strip()]

            # 分发检索 + Small-to-Big 父子块映射
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

            # LLM 轻量级重排
            rerank_prompt = f"以下是从文档中粗筛出的几个片段。请挑选出与问题【{query}】最相关的片段并拼接，剔除无关片段。\n\n片段：\n{unique_parents[:8]}"
            result = agent_self.fast_llm.invoke(rerank_prompt).content
            agent_self._last_retrieved_docs = result  # 供幻觉检测使用
            return result

        @tool
        def recall_memory(query: str) -> str:
            """从用户的历史笔记和对话记录中检索相关记忆。当用户询问之前的笔记、历史对话时使用。"""
            print(f"[TOOL] recall_memory: {query}")
            filter_kwargs = {"filter": {"must": [{"key": "user_id", "match": {"value": agent_self.user_id}}]}}
            try:
                docs = agent_self.memory_store.similarity_search(query, k=4, **filter_kwargs)
            except Exception:
                docs = agent_self.memory_store.similarity_search(query, k=4)
            return "\n\n".join([f"[{d.metadata.get('type', 'note')}]: {d.page_content}" for d in docs])

        @tool
        def add_note(content: str) -> str:
            """将用户提供的内容保存为学习笔记到记忆库。"""
            print(f"[TOOL] add_note: {content[:50]}")
            doc = Document(
                page_content=content,
                metadata={"user_id": agent_self.user_id, "type": "note", "concept": "general"}
            )
            agent_self.memory_store.add_documents([doc])
            agent_self.stats["notes_added"] += 1
            return f"笔记已保存：{content[:50]}..."

        @tool
        def get_stats() -> str:
            """获取当前会话的学习统计信息，包括加载文档数、提问次数、笔记数量等。"""
            s = agent_self.stats
            duration = (datetime.now() - s["session_start"]).seconds
            return (f"会话时长: {duration}秒 | 加载文档: {s['docs_loaded']} | "
                    f"提问次数: {s['questions_asked']} | 笔记数量: {s['notes_added']}")

        return [search_pdf, recall_memory, add_note, get_stats]

    # ---------------------------------------------------------
    # [优化 #1] 手动构建 ReAct 图（Reasoning + Acting 循环）
    # ---------------------------------------------------------
    def _build_agent(self):
        tools = self._build_tools()
        llm_with_tools = self.llm.bind_tools(tools)
        tool_node = ToolNode(tools)

        system_msg = SystemMessage(content=(
            "你是一个专业的学术文档学习助手。你拥有以下工具：\n"
            "- search_pdf：在已加载的 PDF 文档中检索内容\n"
            "- recall_memory：从用户历史笔记和对话中检索记忆\n"
            "- add_note：保存学习笔记\n"
            "- get_stats：查看学习统计\n\n"
            "请根据用户意图自主决定调用哪个工具，或直接回答无需工具的问题。"
            "回答学术问题时务必忠于文档原文，不知则说不知。"
        ))

        # agent 节点：LLM 推理，决定调用工具还是直接回答
        def agent_node(state: MessagesState):
            messages = [system_msg] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # 条件路由：有 tool_calls → 执行工具；否则 → 结束
        def should_continue(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")  # 工具执行完回到 agent，形成 ReAct 循环

        return workflow.compile(checkpointer=self.checkpointer)

    # ==========================================
    # 文档入库 (PyMuPDF4LLM + 父子块)
    # ==========================================
    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}

        start_time = time.time()
        try:
            print(f"正在使用 PyMuPDF4LLM 高精度解析: {pdf_path}")
            md_text = pymupdf4llm.to_markdown(pdf_path)

            # 定义父块（大段落，保留全量上下文）与子块（小段落，用来做高精度向量匹配）
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

            parent_docs = parent_splitter.create_documents([md_text])
            child_docs = []

            print(f"开始构建父子结构，生成父块 {len(parent_docs)} 个...")
            for p_doc in parent_docs:
                p_id = str(uuid.uuid4())
                p_text = p_doc.page_content

                # 在大块内切小块
                c_docs = child_splitter.create_documents([p_text])
                for c in c_docs:
                    # 【核心】把父块的原文和 ID 作为元数据强行塞入小块里
                    c.metadata = {
                        "parent_id": p_id,
                        "parent_text": p_text,
                        "source": os.path.basename(pdf_path),
                        "user_id": self.user_id
                    }
                    child_docs.append(c)

            print(f"入库 {len(child_docs)} 个精确子块向量...")
            self.pdf_store.add_documents(child_docs)

            self.current_document = os.path.basename(pdf_path)
            self.stats["docs_loaded"] += 1

            process_time = time.time() - start_time
            return {"success": True,
                    "message": f"父子块解析成功 (耗时: {process_time:.1f}s)，入库 {len(child_docs)} 个子块向量",
                    "document": self.current_document}

        except Exception as e:
            return {"success": False, "message": str(e)}

    # --- 外部调用的接口保持不变 ---
    def ask(self, question: str) -> str:
        self.stats["questions_asked"] += 1
        config = {"configurable": {"thread_id": self.session_id}}
        original_question = question

        # [优化 #1] ReAct Agent 自主决定调用哪个工具
        # [优化 #2] 幻觉检测反思循环，最多重试 3 次
        for retry in range(4):
            self._last_retrieved_docs = ""
            result = self.app.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
            answer = result["messages"][-1].content

            # 只有调用了 search_pdf 工具才做幻觉检测
            if not self._last_retrieved_docs:
                return answer

            verify_prompt = (
                f"请作为客观的裁判，检查【回答】是否超出了【参考上下文】的范围。\n"
                f"上下文：{self._last_retrieved_docs}\n"
                f"回答：{answer}\n"
                f"如果回答中包含上下文中根本没有提到的实体、数据或硬事实，请回复'FAIL'。如果完全基于上下文，回复'PASS'。"
            )
            check = self.fast_llm.invoke(verify_prompt).content
            hallucination = "FAIL" in check.upper()
            print(f"[INFO] 幻觉检测：{'FAIL' if hallucination else 'PASS'} (retry={retry})")

            if not hallucination or retry >= 3:
                if hallucination:
                    answer += "\n\n⚠️ **[系统校验提示]**：本回答部分内容可能未在当前文档的检索范围内明确提及，存在外部大模型知识（幻觉）的介入，请谨慎参考。"
                # 记录 QA 历史到记忆库
                self.memory_store.add_documents([Document(
                    page_content=f"问题: {original_question}\n回答: {answer}",
                    metadata={"user_id": self.user_id, "type": "qa_history"}
                )])
                return answer

            # 重写问题，下一轮重新检索
            rewrite_prompt = (
                f"原问题检索效果不佳，请重写以下问题使其更适合在学术文档中检索，"
                f"要求更具体、使用专业术语：\n{original_question}"
            )
            question = self.fast_llm.invoke(rewrite_prompt).content.strip()
            print(f"[INFO] 重写后问题 (retry={retry+1})：{question}")

        return answer

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
        # 替换为新的 Industrial 代理
        assistant_state["assistant"] = IndustrialPDFLearningAgent(user_id=user_id)
        return f"✅ 工业级助手已初始化 (用户: {user_id} | 引擎: LangGraph+Qdrant)"

    def reset_db_ui() -> str:
        """调用后端重置数据库方法 (评测沙箱隔离)"""
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        try:
            # 需要确保你的 Agent 类中添加了上文提到的 reset_sandbox 方法
            assistant_state["assistant"].reset_sandbox()
            return "✅ 数据库已彻底清空，评测沙箱已重置！"
        except AttributeError:
            return "❌ 尚未在后端配置 reset_sandbox 方法。"
        except Exception as e:
            return f"❌ 清理失败: {str(e)}"

    def load_pdf(pdf_file) -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        if pdf_file is None:
            return "❌ 请上传PDF文件"

        pdf_path = pdf_file.name
        result = assistant_state["assistant"].load_document(pdf_path)

        if result["success"]:
            return f"✅ {result['message']}\n📄 文档: {result['document']}"
        else:
            return f"❌ {result['message']}"

    def chat(message: str, history: List) -> Tuple[str, List]:
        if assistant_state["assistant"] is None:
            return "", history + [[message, "❌ 请先初始化助手并加载文档"]]
        if not message.strip():
            return "", history

        # 【核心优化】
        # 以前需要用 if any(...) 判断关键词调用 recall
        # 现在底层 LangGraph 已经具备了 LLM 智能分类路由能力
        # 前端直接无脑调用 ask() 即可，Agent 会自己决定是查 PDF、查记忆还是闲聊！
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

    def get_stats_ui() -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        stats = assistant_state["assistant"].get_stats()
        result = "📊 **学习统计**\n\n"
        for key, value in stats.items():
            result += f"- **{key}**: {value}\n"
        return result

    def generate_report_ui() -> str:
        if assistant_state["assistant"] is None:
            return "❌ 请先初始化助手"
        report = assistant_state["assistant"].generate_report(save_to_file=True)
        result = f"✅ 学习报告已生成\n\n**会话信息**\n"
        result += f"- 会话时长: {report['session_info']['duration_seconds']:.0f}秒\n"
        result += f"- 加载文档: {report['learning_metrics']['documents_loaded']}\n"
        result += f"- 提问次数: {report['learning_metrics']['questions_asked']}\n"
        result += f"- 学习笔记: {report['learning_metrics']['concepts_learned']}\n"
        if "report_file" in report:
            result += f"\n💾 报告已保存至: {report['report_file']}"
        return result

    # 构建UI组件...
    with gr.Blocks(title="智能文档问答助手 (LangGraph版)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 智能文档问答助手 (企业级高阶优化版)")

        with gr.Tab("🏠 开始使用"):
            with gr.Row():
                user_id_input = gr.Textbox(label="用户ID", value="pdf_user1", scale=3)
                init_btn = gr.Button("初始化助手", variant="primary", scale=1)
                reset_btn = gr.Button("🗑️ 清空/重置数据库", variant="stop", scale=1)  # 新增按钮

            init_output = gr.Textbox(label="系统状态", interactive=False)

            init_btn.click(init_assistant, inputs=[user_id_input], outputs=[init_output])
            reset_btn.click(reset_db_ui, outputs=[init_output])

            # 修正了文案，体现新的底层技术
            gr.Markdown("### 📄 加载PDF文档 (基于 PyMuPDF4LLM + 父子块检索)")
            pdf_upload = gr.File(label="上传PDF文件", file_types=[".pdf"], type="filepath")
            load_btn = gr.Button("加载文档", variant="primary")
            load_output = gr.Textbox(label="加载状态", interactive=False)
            load_btn.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])

        with gr.Tab("💬 智能问答"):
            chatbot = gr.Chatbot(label="对话历史 (支持查询文档、查询记忆、日常闲聊)", height=400)
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

        with gr.Tab("📊 学习统计"):
            stats_btn = gr.Button("刷新统计", variant="primary")
            stats_output = gr.Markdown()
            stats_btn.click(get_stats_ui, outputs=[stats_output])
            report_btn = gr.Button("生成报告", variant="primary")
            report_output = gr.Textbox(label="报告状态", interactive=False)
            report_btn.click(generate_report_ui, outputs=[report_output])

    return demo

def main():
    print("=" * 60)
    print("正在启动基于 LangGraph + Qdrant 的 Web 界面...")
    print("请确保已配置 OPENAI_API_KEY 环境变量。")
    print("=" * 60)
    demo = create_gradio_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()