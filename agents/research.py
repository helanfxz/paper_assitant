# agents/research.py
# 定义 ResearchAgent 的工具和系统提示。
# list_documents：列出知识库中已入库的所有文档，让 Agent 知道可以查哪些资料。
# search_pdf：通过多查询扩展 + 父子块检索 + LLM 重排，检索与问题最相关的内容，
#             支持按文件名过滤，用于定向查询某篇论文。

from langchain_core.tools import tool
from agents.base import build_sub_agent
from document.registry import format_doc_list
from concurrent.futures import ThreadPoolExecutor

SYSTEM_PROMPT = (
    "你是专业的学术文档检索专家。\n"
    "【工作流程】：\n"
    "1. 如果不确定知识库中有哪些文档，先调用 list_documents 查看\n"
    "2. 使用 search_pdf 检索相关内容，可通过 source 参数指定只在某篇文档中检索\n"
    "3. 基于检索结果忠实回答，不知则说不知\n"
    "【核心纪律】：只针对用户的最新提问进行回答，不重复回答历史问题。"
)


def make_research_agent(llm, fast_llm, pdf_store, loaded_docs: list):

    @tool
    def list_documents() -> str:
        """列出知识库中所有已入库的文档，包含标题、入库时间和摘要。在回答涉及文档内容的问题前可先调用此工具。"""
        result = format_doc_list()

        # documents.json 为空时，直接扫描 Qdrant 获取已有文档的文件名列表
        if "暂无文档" in result:
            try:
                sources = set()
                offset = None
                while True:
                    points, offset = pdf_store.client.scroll(
                        collection_name="pdf_knowledge",
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for p in points:
                        src = (p.payload or {}).get("metadata", {}).get("source") or \
                              (p.payload or {}).get("source", "")
                        if src:
                            sources.add(src)
                    if offset is None:
                        break
                if sources:
                    doc_list = "\n".join(f"- {s}" for s in sorted(sources))
                    result = (
                        f"知识库中检测到 {len(sources)} 个文档（元数据未完整注册，仅显示文件名）：\n{doc_list}\n\n"
                        f"提示：重新上传这些文档可生成标题和摘要。"
                    )
            except Exception as e:
                print(f"  [ResearchAgent][TOOL] Qdrant 扫描失败: {e}")

        print(f"  [ResearchAgent][TOOL] list_documents: {result[:100]}...")
        return result

    @tool
    def search_pdf(query: str, source: str = "") -> str:
        """在已加载的 PDF 文档中检索与问题相关的内容。
        source: 可选，指定文件名（如 'paper.pdf'）只在该文档中检索；留空则搜索全部文档。
        """
        print(f"  [ResearchAgent][TOOL] search_pdf 开始，query: {query}, source: {source or '全部'}")
        print(f"  [ResearchAgent][TOOL] 生成扩展查询词...")
        mqe_prompt = (
            f"为了在学术文档中全面检索以下问题，请生成3个不同表达或侧重点的相似搜索词。"
            f"不要加序号，每行一个。\n原始问题：{query}"
        )
        extended = fast_llm.invoke(mqe_prompt).content.split("\n")
        search_queries = [query] + [q.strip() for q in extended if q.strip()]
        print(f"  [ResearchAgent][TOOL] 扩展查询词: {search_queries}")

        print(f"  [ResearchAgent][TOOL] 向量检索中...")

        def search_one(q):
            if source:
                return pdf_store.similarity_search(
                    q, k=3,
                    filter={"must": [{"key": "source", "match": {"value": source}}]}
                )
            return pdf_store.similarity_search(q, k=3)

        with ThreadPoolExecutor() as executor:
            batches = list(executor.map(search_one, search_queries))
        all_child_docs = [doc for batch in batches for doc in batch]
        print(f"  [ResearchAgent][TOOL] 召回子块数: {len(all_child_docs)}")

        parent_map = {}
        for doc in all_child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in parent_map:
                parent_map[pid] = doc.metadata.get("parent_text", doc.page_content)

        unique_parents = list(parent_map.values())
        if not unique_parents:
            return f"未能在{'「' + source + '」' if source else '知识库'}中检索到相关内容。"

        print(f"  [ResearchAgent][TOOL] 去重后父块数: {len(unique_parents)}，重排中...")
        rerank_prompt = (
            f"以下是从文档中粗筛出的几个片段。请挑选出与问题【{query}】最相关的片段并拼接，"
            f"剔除无关片段。\n\n片段：\n{unique_parents[:8]}"
        )
        result = fast_llm.invoke(rerank_prompt).content
        print(f"  [ResearchAgent][TOOL] 检索完成，结果长度: {len(result)} 字")
        return result

    return build_sub_agent(
        llm, [list_documents, search_pdf], SYSTEM_PROMPT,
        name="ResearchAgent", max_tool_calls=3
    )
