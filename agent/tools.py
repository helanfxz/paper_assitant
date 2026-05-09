"""运行时工具层。

这个文件只负责给主模型暴露可调用工具，不负责主循环编排。
当前主要覆盖三类能力：
1. 文档检索：列文档、混合检索 PDF
2. 记忆与笔记：召回会话摘要、管理研究笔记
3. 运行时状态：返回当前会话统计信息
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent.error_recovery import ModelRecoveryManager
from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.document import format_doc_list
from agent.memory import (
    LEGACY_AUTO_FACT_TYPE,
    PDF_COLLECTION_NAME,
    SESSION_SUMMARY_TYPE,
    delete_study_note,
    list_study_notes,
    save_study_note,
    search_study_notes,
    update_study_note,
)
from agent.skill import SkillLoader

SESSION_MEMORY_RECALL_THRESHOLD = 0.45  # 会话摘要最低召回阈值，过低结果直接丢弃。
MQE_EXPAND_COUNT = 3  # 每次最多生成 3 条扩展查询。
MQE_MIN_QUERY_LEN = 4  # 过短查询不做 MQE，避免扩展出无意义短语。
MQE_DEDUP_RATIO = 0.8  # 简单去重阈值，避免扩展结果只是原查询的小改写。
VECTOR_TOP_K = 3  # 每条扩展查询的向量候选数量。
BM25_TOP_K = 3  # 每条扩展查询的词法候选数量。
RRF_K = 60  # Reciprocal Rank Fusion 平滑参数。
RERANK_PARENT_LIMIT = 8  # 进入最终 rerank 的父块上限。


class QueryExpansionResult(BaseModel):
    """约束 MQE 输出，避免小模型返回多余解释。"""

    queries: list[str] = Field(default_factory=list, min_length=1, max_length=MQE_EXPAND_COUNT)


def _normalize_todo_status(status: str) -> str:
    """把 todo 状态收紧到系统支持的固定集合。"""
    cleaned_status = str(status).strip().lower()
    if cleaned_status in {"todo", "doing", "done"}:
        return cleaned_status
    return "todo"


def _build_time_scope(
    kind: str = "",
    start_at: str = "",
    end_at: str = "",
) -> dict[str, str]:
    """构建统一的时间范围结构。"""
    cleaned_kind = str(kind).strip().lower()
    cleaned_start = str(start_at).strip()
    cleaned_end = str(end_at).strip()

    if cleaned_kind not in {"none", "deadline", "window"}:
        if cleaned_start and cleaned_end:
            cleaned_kind = "window"
        elif cleaned_end:
            cleaned_kind = "deadline"
        else:
            cleaned_kind = "none"

    return {
        "kind": cleaned_kind,
        "start_at": cleaned_start,
        "end_at": cleaned_end,
    }


def _format_time_scope(time_scope: dict[str, Any] | None) -> str:
    """把时间范围转成可读文本。"""
    time_scope = time_scope or {}
    kind = str(time_scope.get("kind", "none")).strip().lower()
    start_at = str(time_scope.get("start_at", "")).strip()
    end_at = str(time_scope.get("end_at", "")).strip()

    if kind == "deadline" and end_at:
        return f"截止时间：{end_at}"
    if kind == "window":
        if start_at and end_at:
            return f"时间范围：{start_at} ~ {end_at}"
        if end_at:
            return f"结束时间：{end_at}"
        if start_at:
            return f"开始时间：{start_at}"
    return "时间范围：未设置"


def _doc_key(doc: Document) -> str:
    """给 child chunk 生成稳定 key，便于多路检索结果去重。"""
    chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return chunk_id
    return f"{doc.metadata.get('source', '')}|{doc.metadata.get('parent_id', '')}|{doc.page_content[:80]}"


def _deduplicate_queries(queries: list[str]) -> list[str]:
    """去掉重复或近似重复查询，避免把 rerank 预算浪费在等价查询上。"""
    unique_queries: list[str] = []
    seen_normalized: list[str] = []

    for raw_query in queries:
        cleaned_query = " ".join(str(raw_query).strip().split())
        if not cleaned_query:
            continue

        normalized_query = cleaned_query.lower()
        keep_query = True

        # 这里故意只做很轻的字符串相似过滤。
        # 查询扩展的目标是补视角，不是做复杂文本聚类。
        for seen_query in seen_normalized:
            overlap = sum(1 for char in normalized_query if char in seen_query)
            overlap_ratio = overlap / max(len(normalized_query), len(seen_query), 1)
            if normalized_query == seen_query or overlap_ratio >= MQE_DEDUP_RATIO:
                keep_query = False
                break

        if keep_query:
            unique_queries.append(cleaned_query)
            seen_normalized.append(normalized_query)

        if len(unique_queries) >= MQE_EXPAND_COUNT:
            break

    return unique_queries


def _expand_queries(query: str, fast_llm, recovery_manager: ModelRecoveryManager) -> list[str]:
    """用 few-shot MQE 生成补充查询，提升混合检索的召回覆盖面。"""
    cleaned_query = " ".join(query.strip().split())
    if len(cleaned_query) < MQE_MIN_QUERY_LEN:
        return [cleaned_query]

    mqe_prompt = (
        "你在为学术论文检索生成查询扩展。目标是补充不同检索角度，不是简单改写原句。\n"
        "要求：\n"
        "1. 返回 1 到 3 条 query。\n"
        "2. 优先覆盖方法名、任务名、机制、应用场景、评价维度等不同角度。\n"
        "3. 不要输出解释、编号或多余文本。\n"
        "4. 每条 query 要简洁，可直接用于向量检索和关键词检索。\n\n"
        "示例 1\n"
        "用户问题：LoRA 为什么参数高效？\n"
        "输出：\n"
        "- LoRA 参数高效 低秩分解 原理\n"
        "- LoRA fine-tuning parameter efficiency low-rank adaptation\n"
        "- LoRA 与全量微调 参数量 训练开销 对比\n\n"
        "示例 2\n"
        "用户问题：这篇论文怎么做多模态对齐？\n"
        "输出：\n"
        "- 多模态对齐 方法 视觉文本 表示学习\n"
        "- vision-language alignment mechanism cross-modal learning\n"
        "- 图文对齐 训练目标 对比学习 匹配损失\n\n"
        f"用户问题：{cleaned_query}"
    )

    expansion_result = recovery_manager.invoke_structured_model(
        fast_llm,
        QueryExpansionResult,
        mqe_prompt,
        purpose="检索查询扩展",
    )
    if expansion_result.ok:
        candidate_queries = [cleaned_query, *expansion_result.value.queries]
    else:
        candidate_queries = [cleaned_query]

    deduplicated_queries = _deduplicate_queries(candidate_queries)
    return deduplicated_queries or [cleaned_query]


def _rrf_fuse_ranked_lists(ranked_lists: list[list[Document]]) -> list[tuple[Document, float]]:
    """用 RRF 融合多路排序结果，避免强行对齐不同检索器的分数尺度。"""
    fused_scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for ranked_docs in ranked_lists:
        for rank_index, doc in enumerate(ranked_docs):
            doc_key = _doc_key(doc)
            docs_by_key.setdefault(doc_key, doc)
            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (RRF_K + rank_index + 1)

    ranked_pairs = sorted(
        ((docs_by_key[doc_key], score) for doc_key, score in fused_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked_pairs


def build_runtime_tools(
    fast_llm,
    rerank_llm,
    recovery_manager: ModelRecoveryManager,
    pdf_store,
    lexical_index,
    session_memory_store,
    study_notes_store,
    preference_store,
    user_id: str,
    session_id: str,
    session_stats: dict,
    runtime,
    skill_loader: SkillLoader,
):
    """构建主模型当前会话可调用的工具列表。"""

    @tool
    def load_skill(name: str) -> str:
        """按名称加载完整 skill 正文；只有当前任务需要专项流程时才调用。"""
        return skill_loader.load_skill(name)

    @tool
    def list_documents() -> str:
        """列出当前知识库中的论文文档，帮助模型确认有哪些材料可检索。"""
        formatted_list = format_doc_list()
        if "暂无文档" not in formatted_list:
            return formatted_list

        # documents.json 为空时，退化到直接扫描 Qdrant，至少能给模型一个可见文档清单。
        sources: set[str] = set()
        offset = None
        while True:
            points, offset = pdf_store.client.scroll(
                collection_name=PDF_COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload if isinstance(point.payload, dict) else {}
                metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else payload
                source_name = str(metadata.get("source", "")).strip()
                if source_name:
                    sources.add(source_name)
            if offset is None:
                break

        if not sources:
            return "知识库中暂无文档，请先上传 PDF 文件。"

        doc_lines = [f"知识库中共有 {len(sources)} 篇文档：", ""]
        for index, source_name in enumerate(sorted(sources), 1):
            doc_lines.append(f"{index}. {source_name}")
        return "\n".join(doc_lines)

    @tool
    def search_pdf(query: str, source: str = "") -> str:
        """在当前用户的论文库中检索相关内容，返回最相关的父块片段。"""
        cleaned_query = " ".join(query.strip().split())
        cleaned_source = source.strip()
        if not cleaned_query:
            return "检索失败：query 不能为空。"

        expanded_queries = _expand_queries(cleaned_query, fast_llm, recovery_manager)

        lexical_search_enabled = True
        # 先确保旧文档也已经补建 lexical index，避免新老数据检索表现不一致。
        # 如果这一步失败，不应该让整条查询链路直接崩掉，先退化成纯向量检索。
        try:
            lexical_index.ensure_source_indexed_from_vector_store(
                pdf_store,
                user_id=user_id,
                source=cleaned_source,
            )
        except Exception as exc:
            lexical_search_enabled = False
            print(f"[Retrieval] 词法索引回补失败，退化为纯向量检索: {exc}")

        vector_filter = None
        if cleaned_source:
            vector_filter = {"must": [{"key": "source", "match": {"value": cleaned_source}}]}

        def vector_search(single_query: str) -> list[Document]:
            return pdf_store.similarity_search(single_query, k=VECTOR_TOP_K, filter=vector_filter)

        def lexical_search(single_query: str) -> list[Document]:
            if not lexical_search_enabled:
                return []
            return lexical_index.search(
                single_query,
                user_id=user_id,
                source=cleaned_source,
                top_k=BM25_TOP_K,
            )

        # 多条扩展查询是独立检索任务，这里并发执行可以直接降低单次查询时延。
        ranked_lists: list[list[Document]] = []
        failed_routes = 0
        with ThreadPoolExecutor(max_workers=max(2, len(expanded_queries) * 2)) as executor:
            vector_futures = [executor.submit(vector_search, single_query) for single_query in expanded_queries]
            lexical_futures = (
                [executor.submit(lexical_search, single_query) for single_query in expanded_queries]
                if lexical_search_enabled
                else []
            )

            for future in vector_futures + lexical_futures:
                try:
                    ranked_lists.append(future.result())
                except Exception as exc:
                    failed_routes += 1
                    print(f"[Retrieval] 单路检索失败，已跳过: {exc}")

        fused_ranked_children = _rrf_fuse_ranked_lists(ranked_lists)
        if not fused_ranked_children and failed_routes:
            raise RuntimeError("向量检索与词法检索均未返回可用结果")
        if not fused_ranked_children:
            return "未找到相关内容。"

        # RRF 仍然是在 child chunk 层融合，这一层负责补召回和排序。
        # 最后交给主模型使用的仍然应该是父块，以保留更完整的上下文。
        parent_hits: dict[str, dict[str, object]] = {}
        for child_doc, fused_score in fused_ranked_children:
            parent_id = str(child_doc.metadata.get("parent_id", ""))
            if not parent_id:
                continue

            parent_record = parent_hits.setdefault(
                parent_id,
                {
                    "score": 0.0,
                    "parent_text": str(child_doc.metadata.get("parent_text", child_doc.page_content)),
                    "source": str(child_doc.metadata.get("source", "")),
                },
            )
            parent_record["score"] = float(parent_record["score"]) + fused_score

        if not parent_hits:
            return "未找到相关内容。"

        ranked_parents = sorted(
            parent_hits.items(),
            key=lambda item: float(item[1]["score"]),
            reverse=True,
        )[:RERANK_PARENT_LIMIT]

        candidate_texts = [
            str(parent_record["parent_text"]).strip()
            for _, parent_record in ranked_parents
            if str(parent_record["parent_text"]).strip()
        ]
        if candidate_texts:
            try:
                rerank_hits = rerank_llm.rerank(
                    cleaned_query,
                    candidate_texts,
                    top_n=min(3, len(candidate_texts)),
                )
                reranked_blocks: list[str] = []
                for hit in rerank_hits[:3]:
                    _, parent_record = ranked_parents[hit.index]
                    source_name = str(parent_record.get("source", "")).strip()
                    parent_text = str(parent_record.get("parent_text", "")).strip()
                    if parent_text:
                        reranked_blocks.append(f"来源文档：{source_name}\n{parent_text}")
                if reranked_blocks:
                    return "\n\n".join(reranked_blocks)
            except Exception as exc:
                print(f"[Retrieval] rerank 失败，使用混合检索结果: {exc}")

        fallback_blocks = [
            str(parent_record["parent_text"]).strip()
            for _, parent_record in ranked_parents[:2]
        ]
        return "\n\n".join(block for block in fallback_blocks if block) or "未找到相关内容。"

    @tool
    def recall_memory(query: str, memory_type: str = "session_summary") -> str:
        """显式查询当前会话的压缩摘要，帮助模型回看超出窗口的历史信息。"""
        cleaned_query = " ".join(query.strip().split())
        if not cleaned_query:
            return "查询失败：query 不能为空。"
        if memory_type not in {"session_summary", "all"}:
            return "查询失败：memory_type 仅支持 session_summary 或 all。"

        # 这里不吞掉底层向量库异常。
        # 如果 Qdrant 查询本身失败，应该交给工具错误恢复层区分为“查询执行失败”，
        # 而不是和“没召回到内容”混成同一种返回。
        recall_results = session_memory_store.similarity_search_with_score(
            cleaned_query,
            k=5,
            filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "session_id", "match": {"value": session_id}},
                ]
            },
        )

        matched_entries: list[str] = []
        for recalled_doc, similarity_score in recall_results:
            entry_type = str(recalled_doc.metadata.get("type", SESSION_SUMMARY_TYPE))
            # 当前 study notes 已经独立，这个工具只负责当前会话摘要。
            # 因此即使 memory_type=all，也只会返回 session summary 相关内容。
            if memory_type != "all" and entry_type not in {
                memory_type,
                SESSION_SUMMARY_TYPE if memory_type == "session_summary" else None,
            }:
                continue
            if entry_type not in {SESSION_SUMMARY_TYPE, LEGACY_AUTO_FACT_TYPE}:
                continue
            if similarity_score < SESSION_MEMORY_RECALL_THRESHOLD:
                continue
            matched_entries.append(f"- {recalled_doc.page_content}")

        if not matched_entries:
            return "未找到相关会话摘要。"
        return "相关会话摘要：\n" + "\n".join(matched_entries)

    @tool
    def save_preference(
        scope: str,
        type: str,
        value: str,
        operation: str = "add",
        source_text: str = "",
    ) -> str:
        """保存跨会话长期偏好。仅在用户明确表达稳定、长期适用的回答习惯时调用；不要为当前这一轮的临时要求调用。
        scope 只能是：global / paper_summary / paper_compare。
        type 只能是：language / format / detail_level / focus / avoid。"""
        preference = {
            "is_preference": True,
            "scope": str(scope).strip(),
            "type": str(type).strip(),
            "value": str(value).strip(),
            "operation": str(operation).strip() or "add",
            "confidence": "high",
            "reason": "由主模型主动发起的长期偏好保存请求",
            "sensitive": str(operation).strip() in {"delete", "clear"},
        }
        return preference_store.apply_preference_update(
            preference,
            source_text=str(source_text).strip(),
            confirmed=True,
        )

    @tool
    def save_todo(
        title: str,
        detail: str = "",
        status: str = "todo",
        time_kind: str = "none",
        start_at: str = "",
        end_at: str = "",
        subtasks: list[str] | None = None,
    ) -> str:
        """为当前 session 保存一条 todo。
        如果任务有多个阶段或子步骤，用 subtasks 传入子任务标题列表，不要拆成多条 todo。
        subtasks 示例：["阶段一：阅读 MoLA", "阶段二：阅读 LoraHub"]"""
        cleaned_title = str(title).strip()
        cleaned_detail = str(detail).strip()
        if not cleaned_title and not cleaned_detail:
            return "保存失败：todo 标题和详情不能同时为空。"

        now = datetime.now().isoformat()
        todo_id = f"todo_{uuid4().hex[:8]}"

        # 把子任务标题列表转成带状态的子任务对象。
        subtask_list: list[dict[str, str]] = []
        for subtask_title in (subtasks or []):
            cleaned = str(subtask_title).strip()
            if cleaned:
                subtask_list.append({"title": cleaned, "status": "todo"})

        todo = {
            "todo_id": todo_id,
            "title": cleaned_title or cleaned_detail[:30],
            "detail": cleaned_detail,
            "status": _normalize_todo_status(status),
            "subtasks": subtask_list,
            "time_scope": _build_time_scope(time_kind, start_at, end_at),
            "created_at": now,
            "updated_at": now,
        }
        runtime.todos.append(todo)
        subtask_info = f"，包含 {len(subtask_list)} 个子任务" if subtask_list else ""
        return f"todo 已保存：{todo['title']}（todo_id={todo_id}）{subtask_info}"

    @tool
    def list_todos() -> str:
        """列出当前 session 的 todos。"""
        todos = list(runtime.todos)
        if not todos:
            return "当前会话没有 todo。"

        lines = [f"当前会话共有 {len(todos)} 条 todo：", ""]
        for index, todo in enumerate(todos, 1):
            lines.append(
                f"{index}. {todo.get('title', '未命名任务')} | "
                f"status={todo.get('status', 'todo')} | "
                f"todo_id={todo.get('todo_id', '')}"
            )
            detail = str(todo.get("detail", "")).strip()
            if detail:
                lines.append(f"   详情：{detail}")
            lines.append(f"   {_format_time_scope(todo.get('time_scope'))}")
            subtasks = todo.get("subtasks") or []
            if subtasks:
                lines.append(f"   子任务（共 {len(subtasks)} 项）：")
                for i, subtask in enumerate(subtasks):
                    lines.append(
                        f"     [{i}] {subtask.get('title', '未命名')} | status={subtask.get('status', 'todo')}"
                    )
        return "\n".join(lines)

    @tool
    def update_todo_status(todo_id: str, status: str, subtask_index: int = -1) -> str:
        """更新当前 session 中某条 todo 的状态。
        如果要更新子任务状态，传入 subtask_index（从 0 开始的索引，可从 list_todos 查看）；
        不传或传 -1 则更新整条 todo 的状态。"""
        cleaned_todo_id = str(todo_id).strip()
        if not cleaned_todo_id:
            return "更新失败：todo_id 不能为空。"

        for todo in runtime.todos:
            if str(todo.get("todo_id", "")).strip() != cleaned_todo_id:
                continue

            now = datetime.now().isoformat()
            # 更新子任务状态。
            if subtask_index >= 0:
                subtasks = todo.get("subtasks") or []
                if subtask_index >= len(subtasks):
                    return f"更新失败：subtask_index={subtask_index} 超出范围，该 todo 共有 {len(subtasks)} 个子任务。"
                subtasks[subtask_index]["status"] = _normalize_todo_status(status)
                todo["updated_at"] = now
                return (
                    f"子任务已更新：[{subtask_index}] {subtasks[subtask_index].get('title', '')} "
                    f"-> {subtasks[subtask_index]['status']}"
                )

            # 更新整条 todo 状态。
            todo["status"] = _normalize_todo_status(status)
            todo["updated_at"] = now
            return f"todo 状态已更新：{todo.get('title', cleaned_todo_id)} -> {todo['status']}"

        return "更新失败：未找到对应 todo。"

    @tool
    def delete_todo(todo_id: str) -> str:
        """删除当前 session 中某条 todo。"""
        cleaned_todo_id = str(todo_id).strip()
        if not cleaned_todo_id:
            return "删除失败：todo_id 不能为空。"

        for index, todo in enumerate(runtime.todos):
            if str(todo.get("todo_id", "")).strip() != cleaned_todo_id:
                continue
            removed_todo = runtime.todos.pop(index)
            return f"todo 已删除：{removed_todo.get('title', cleaned_todo_id)}"
        return "删除失败：未找到对应 todo。"

    @tool
    def save_note(content: str, title: str = "") -> str:
        """保存研究笔记，供后续跨会话回顾和复用。"""
        cleaned_content = content.strip()
        if not cleaned_content:
            return "保存失败：笔记内容不能为空。"

        # 写入类工具默认只执行一次，不在工具外层自动重试。
        # 这里直接返回 note_id，便于后续更新和删除。
        note_id = save_study_note(
            study_notes_store,
            cleaned_content,
            user_id=user_id,
            source_session_id=session_id,
            title=title.strip(),
        )
        session_stats["notes_added"] += 1
        return f"笔记已保存，note_id={note_id}。"

    @tool
    def list_notes() -> str:
        """列出当前用户的全部研究笔记，便于查看和管理。"""
        # 这里直接把底层异常交给工具错误恢复层处理，
        # 这样“读取失败”和“当前没有笔记”就不会混在一起。
        note_entries = list_study_notes(study_notes_store, user_id)
        if not note_entries:
            return "当前没有研究笔记。"

        lines = [f"共有 {len(note_entries)} 条研究笔记：", ""]
        for index, note_entry in enumerate(note_entries, 1):
            title = note_entry.get("title") or note_entry.get("content", "")[:30]
            lines.append(f"{index}. {title} | note_id={note_entry['note_id']}")
        return "\n".join(lines)

    @tool
    def search_notes(query: str) -> str:
        """按语义检索研究笔记，返回最相关的已保存笔记。"""
        cleaned_query = " ".join(query.strip().split())
        if not cleaned_query:
            return "查询失败：query 不能为空。"

        # 研究笔记检索和 recall_memory 一样，
        # 需要区分“底层检索失败”和“没有找到相关笔记”。
        matched_notes = search_study_notes(study_notes_store, user_id, cleaned_query)
        if not matched_notes:
            return "未找到相关研究笔记。"

        lines = ["相关研究笔记：", ""]
        for note_entry in matched_notes:
            lines.append(
                f"[{note_entry['title'] or note_entry['note_id']}] note_id={note_entry['note_id']}\n"
                f"{note_entry['content']}"
            )
        return "\n\n".join(lines)

    @tool
    def update_note(note_id: str, new_content: str, new_title: str = "") -> str:
        """更新已有研究笔记内容。"""
        if not note_id.strip():
            return "更新失败：note_id 不能为空。"
        if not new_content.strip():
            return "更新失败：新内容不能为空。"

        # 当前更新逻辑已经改成同 note_id 覆盖写，
        # 不再先删旧笔记再写新笔记，避免中途失败导致笔记丢失。
        updated = update_study_note(
            study_notes_store,
            user_id=user_id,
            note_id=note_id.strip(),
            new_content=new_content.strip(),
            new_title=new_title.strip(),
        )
        return f"笔记已更新，note_id={note_id.strip()}。" if updated else "更新失败：未找到对应笔记。"

    @tool
    def delete_note(note_id: str) -> str:
        """删除已有研究笔记。"""
        if not note_id.strip():
            return "删除失败：note_id 不能为空。"

        deleted = delete_study_note(study_notes_store, user_id=user_id, note_id=note_id.strip())
        return f"笔记已删除，note_id={note_id.strip()}。" if deleted else "删除失败：未找到对应笔记。"

    @tool
    def get_stats() -> str:
        """返回当前会话的运行统计信息，供模型判断上下文状态。"""
        running_seconds = int((datetime.now() - session_stats["session_start"]).total_seconds())
        return (
            "当前会话统计：\n"
            f"- 已提问轮数：{session_stats.get('questions_asked', 0)}\n"
            f"- 已加载文档数：{session_stats.get('docs_loaded', 0)}\n"
            f"- 已保存笔记数：{session_stats.get('notes_added', 0)}\n"
            f"- 会话已运行秒数：{running_seconds}"
        )

    return [
        load_skill,
        list_documents,
        search_pdf,
        recall_memory,
        save_preference,
        save_todo,
        list_todos,
        update_todo_status,
        delete_todo,
        save_note,
        list_notes,
        search_notes,
        update_note,
        delete_note,
        get_stats,
    ]


make_tools = build_runtime_tools
