# 智能文档问答助手

基于 LangGraph + Qdrant 的企业级文档问答系统，支持 PDF 上传、智能检索、学习记录统计等功能。

---

## 项目简介

用户可上传 PDF 文档，向助手提问文档相关内容；系统同时支持查询历史对话记忆和日常闲聊。助手还提供学习笔记管理与学习统计报告功能。

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 工作流编排 | LangGraph (StateGraph) |
| 向量数据库 | Qdrant |
| LLM / Embedding | 智谱 GLM-5 / GLM-4-Flash / Embedding-3 |
| PDF 解析 | PyMuPDF4LLM |
| 前端 UI | Gradio |
| 文本分割 | LangChain RecursiveCharacterTextSplitter |

---

## 项目结构

```
my_agent_project/
├── app.py              # 主程序（核心逻辑 + Gradio UI）
├── memory_data/        # 持久化记忆数据（无需查看）
├── paper/              # 示例 PDF 文档
├── result/             # 生成的学习报告
└── venv/               # Python 虚拟环境
```

---

## 核心架构

### LangGraph 状态定义（AgentState）

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]   # 对话历史
    user_id: str                  # 用户标识
    query: str                    # 当前问题
    action_type: str              # 路由类型：pdf / memory / general
    extended_queries: List[str]   # 多查询扩展结果
    retrieved_docs: str           # 检索到的文档内容
    retrieved_memory: str         # 检索到的记忆内容
    final_answer: str             # 最终回答
    hallucination_risk: bool      # 幻觉风险标记
```

### 图结构（工作流）

```
START
  └─→ [路由节点] ──→ retrieve_pdf_node   ──┐
                 ├─→ recall_memory_node  ──┤
                 └─→ general_chat_node   ──┤
                                           ↓
                                  generate_answer_node
                                           ↓
                                  verify_hallucination_node
                                           ↓
                                          END
```

---

## 详细流程

### 1. 智能路由

使用快速模型（GLM-4-Flash）对用户输入进行意图分类，输出三种路由：

- `pdf`：学术问题，需查阅文档知识库
- `memory`：查询用户历史笔记或对话记录
- `general`：日常闲聊或基础常识问答

### 2. 检索节点（retrieve_pdf_node）

针对 `pdf` 类型问题，执行以下步骤：

1. **多查询扩展（MQE）**：用快速模型将原始问题扩展为 3 个不同表达的搜索词，加上原始问题共 4 条查询。
2. **子块检索**：每条查询在 Qdrant 中检索 top-3 小块，共最多 12 个候选子块。
3. **父子块映射（Small-to-Big）**：根据子块 metadata 中的 `parent_id` 取出对应父块文本，用哈希表去重，避免同一父块被重复加入。
4. **LLM 轻量重排（Reranking）**：将最多 8 个父块文本交给快速模型，筛选并拼接与问题最相关的片段，作为最终上下文。

### 3. 回忆节点（recall_memory_node）

针对 `memory` 类型问题，按 `user_id` 过滤，在语义记忆库（`user_semantic_memory`）中检索最相似的 4 条历史记录（笔记或 QA 历史）。

### 4. 普通问答节点（general_chat_node）

针对 `general` 类型问题，不做额外检索，直接标记 `action_type`，由生成节点直接调用 LLM 回答。

### 5. 生成回答节点（generate_answer_node）

根据 `action_type` 构造不同的 Prompt，调用主模型（GLM-5）生成回答：

- `pdf`：基于检索到的文档片段回答，要求忠于原文
- `memory`：基于历史记忆回答
- `general`：自然对话

### 6. 幻觉检测节点（verify_hallucination_node）

仅对 `pdf` 类型回答生效：

- 让快速模型判断回答是否超出检索上下文范围
- 若检测到幻觉（`FAIL`），在回答末尾追加警告提示
- 无论是否通过，将本次 QA 记录存入语义记忆库，供后续 `memory` 查询使用

> 注：当前代码中幻觉检测后直接结束，介绍中提到的"重写问题重试（最多3次）"为设计规划，尚未在代码中实现。

---

## 文档入库流程

使用 `load_document(pdf_path)` 方法：

1. **PDF 解析**：通过 `pymupdf4llm.to_markdown()` 将 PDF 转为 Markdown 文本，保留结构信息。
2. **父块切分**：`chunk_size=1500, overlap=200`，生成语义完整的大段落（父块）。
3. **子块切分**：对每个父块再用 `chunk_size=400, overlap=50` 切分为小块（子块）。
4. **元数据注入**：每个子块的 metadata 包含：
   - `parent_id`：父块唯一 ID（UUID）
   - `parent_text`：父块原文（用于 Small-to-Big 检索回溯）
   - `source`：文件名
   - `user_id`：用户 ID
5. **入库**：仅将子块向量化存入 Qdrant `pdf_knowledge` 集合，检索时通过 metadata 回溯父块内容。

---

## Qdrant 集合说明

| 集合名 | 用途 | 向量维度 |
|--------|------|----------|
| `pdf_knowledge` | 存储文档子块向量 | 2048 |
| `user_semantic_memory` | 存储用户笔记和 QA 历史 | 2048 |

---

## Gradio UI 功能

| Tab | 功能 |
|-----|------|
| 开始使用 | 初始化助手（输入用户ID）、上传并加载 PDF、重置数据库 |
| 智能问答 | 多轮对话，自动路由到文档检索 / 记忆查询 / 闲聊 |
| 学习笔记 | 手动添加笔记，关联概念标签，存入语义记忆库 |
| 学习统计 | 查看会话统计数据，生成并保存学习报告 |

---

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（.env 文件）
ZHIPU_API_KEY=your_key
ZHIPU_URL=https://open.bigmodel.cn/api/paas/v4/
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key  # 可选

# 3. 启动服务
python app.py
# 访问 http://localhost:7860
```
