# 优化计划（面向 Agent 开发岗位）

> 目标：提升项目的 Agent 技术深度，重点体现 ReAct、Reflection、Multi-Agent、记忆系统等核心能力，为 Agent 开发岗位面试做准备。

---

## 优先级总览

| 优先级 | 编号 | 优化点 | 类别 | 状态 |
|--------|------|--------|------|------|
| 🔴 高 | #1 | ReAct / Tool-Use Agent 改造 | Agent 核心 | 待实现 |
| 🔴 高 | #2 | 幻觉检测反思循环（Self-Reflection） | Agent 核心 | 待实现 |
| 🟡 中 | #3 | Planning 节点（任务分解） | Agent 核心 | 待实现 |
| 🟡 中 | #4 | Multi-Agent 协作架构 | Agent 核心 | 待实现 |
| 🟡 中 | #5 | 长期记忆分层管理 | Agent 核心 | 待实现 |
| 🟢 低 | #6 | 混合检索（Hybrid Search） | RAG 检索 | 待实现 |
| 🟢 低 | #7 | 专用 Reranker 模型 | RAG 检索 | 待实现 |
| 🟢 低 | #8 | 流式输出（Streaming） | 工程质量 | 待实现 |
| 🟢 低 | #9 | 异步化（Async） | 工程质量 | 待实现 |
| 🟢 低 | #10 | 可观测性（LangSmith/Langfuse） | 工程质量 | 待实现 |

---

## Agent 核心能力优化（重点）

### #1 ReAct / Tool-Use Agent 改造

**现状**：固定线性图，路由后直接执行，LLM 没有工具调用能力，不是真正的 Agent。

**目标**：将 Agent 改造为 ReAct 模式，LLM 自主决定调用哪个工具、何时停止推理。

**实现思路**：
```python
from langgraph.prebuilt import create_react_agent

tools = [search_pdf_tool, recall_memory_tool, add_note_tool, get_stats_tool]
agent = create_react_agent(llm, tools, checkpointer=checkpointer)
```

**面试价值**：ReAct 是 Agent 开发的核心范式，体现对 Reasoning + Acting 论文的工程落地能力。

---

### #2 幻觉检测反思循环（Self-Reflection）【最优先实现】

**现状**：`_verify_hallucination_node` 检测到幻觉只追加警告文字，没有重试逻辑（代码与设计文档不符）。

**目标**：实现完整的 Reflection Loop，检测失败时重写问题重新检索，最多重试 3 次。

**实现思路**：
```python
# 在 AgentState 中增加字段
retry_count: int  # 重试次数

# 新增重写节点
def _rewrite_query_node(self, state: AgentState) -> AgentState:
    rewrite_prompt = f"原问题检索效果不佳，请重写以下问题使其更适合文档检索：{state['query']}"
    new_query = self.fast_llm.invoke(rewrite_prompt).content
    return {"query": new_query, "retry_count": state.get("retry_count", 0) + 1}

# 幻觉检测后的条件路由
def _route_after_verify(self, state: AgentState) -> str:
    if state["hallucination_risk"] and state.get("retry_count", 0) < 3:
        return "rewrite_query_node"
    return END

workflow.add_conditional_edges("verify_hallucination_node", _route_after_verify)
workflow.add_edge("rewrite_query_node", "retrieve_pdf_node")
```

**面试价值**：对应 Self-RAG、CRAG 等论文的工程落地，是 Agentic RAG 的标志性设计。

---

### #3 Planning 节点（任务分解）

**现状**：每次只处理单轮问题，无法处理"比较 A 和 B 的区别并总结优缺点"这类复杂多步骤任务。

**目标**：对复杂问题先做 Plan，拆解为子任务后逐步执行，最后汇总。

**实现思路**：
```python
def _plan_node(self, state: AgentState) -> AgentState:
    plan_prompt = f"判断以下问题是否需要分解为子问题。如需要，输出 2-4 个子问题，每行一个；否则输出 SIMPLE。\n问题：{state['query']}"
    result = self.llm.invoke(plan_prompt).content
    if result.strip() != "SIMPLE":
        sub_tasks = [q.strip() for q in result.split("\n") if q.strip()]
        return {"sub_tasks": sub_tasks, "current_step": 0}
    return {"sub_tasks": [], "current_step": 0}
```

**面试价值**：对应 Plan-and-Execute、LLM Compiler 等 Agent 架构。

---

### #4 Multi-Agent 协作架构

**现状**：单一 Agent 处理所有任务，职责不清晰，难以扩展。

**目标**：拆分为 Supervisor + 专家 Agent 架构，各 Agent 职责单一。

**架构设计**：
```
SupervisorAgent
├── ResearchAgent   # 负责文档检索与问答
├── SummaryAgent    # 负责内容总结与报告生成
└── MemoryAgent     # 负责记忆存取与笔记管理
```

**实现思路**：
```python
from langgraph_supervisor import create_supervisor

research_agent = create_react_agent(llm, [search_pdf_tool], name="ResearchAgent")
memory_agent = create_react_agent(llm, [recall_tool, add_note_tool], name="MemoryAgent")
supervisor = create_supervisor([research_agent, memory_agent], model=llm)
```

**面试价值**：Multi-Agent 是当前 Agent 领域最热方向，LangGraph 的核心卖点，面试必谈。

---

### #5 长期记忆分层管理

**现状**：所有记忆混存在一个 Qdrant 集合，无分层，无遗忘机制。

**目标**：参考 MemGPT 设计三层记忆架构。

**架构设计**：

| 层级 | 内容 | 存储方案 |
|------|------|----------|
| 工作记忆（Working Memory） | 当前对话上下文 | LangGraph State（现有） |
| 情节记忆（Episodic Memory） | QA 历史、用户笔记 | Qdrant `user_semantic_memory`（现有） |
| 语义记忆（Semantic Memory） | 用户知识图谱、概念关系 | Neo4j 图数据库（待引入） |

**面试价值**：体现对 Agent 记忆系统的深度理解，区别于普通 RAG 开发者。

---

## RAG 检索优化

### #6 混合检索（Hybrid Search）

**现状**：纯向量检索，对精确词汇（人名、公式、专有缩写）效果差。

**目标**：向量检索（Dense）+ BM25 稀疏检索（Sparse）融合，Qdrant 原生支持。

```python
from qdrant_client.models import SparseVector, NamedSparseVector
# 配置 sparse + dense 双向量检索
```

---

### #7 替换 LLM Reranker 为专用模型

**现状**：用 GLM-4-Flash 做重排，成本高、延迟大、不专业。

**目标**：接入 BGE-Reranker-v2 或 Cohere Rerank API，精度更高、延迟更低。

---

## 工程质量优化

### #8 流式输出（Streaming）

**现状**：`llm.invoke()` 阻塞等待，用户需等待完整回答才能看到内容。

**目标**：改用 `llm.astream()` + Gradio 流式组件，实现打字机效果。

---

### #9 异步化（Async）

**现状**：多查询扩展的 4 条检索串行执行，延迟叠加。

**目标**：用 `asyncio.gather` 并发执行多条检索。

```python
import asyncio
results = await asyncio.gather(*[
    self.pdf_store.asimilarity_search(q, k=3) for q in search_queries
])
```

---

### #10 可观测性（Observability）

**现状**：只有 `print("[INFO]...")` 日志，无法追踪 Agent 运行细节。

**目标**：接入 LangSmith 或 Langfuse，追踪每次运行的完整 trace（节点输入输出、token 消耗、延迟）。

```python
# LangSmith 只需设置环境变量即可自动追踪
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
```

**面试价值**：体现生产化意识，是区分初级和中高级 Agent 开发者的关键指标。

---

## 面试准备建议

1. **优先实现 #2**：代码改动小，但技术含量高，能直接展示 Agentic RAG 的核心设计能力。
2. **优先实现 #1**：将固定流程改为 Tool-Use，是从"RAG 应用"升级为"真正 Agent"的关键一步。
3. **口头描述 #4**：Multi-Agent 架构可作为"下一步规划"在面试中阐述，展示技术视野不需要完全实现。
4. **准备对比说明**：每个优化点都要能说清楚"改之前是什么、改之后是什么、为什么这样改"。
