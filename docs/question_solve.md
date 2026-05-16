# 项目痛点与解决方案记录

---

## 痛点 1：按页切块导致跨页段落语义断裂

### 背景

当前方案为了确保 chunk 的 `page_number` 绝对精确，每页独立做 parent/child split，禁止先拼整篇 Markdown 再跨页切分。

```
Page N 文本 → parent chunks → child chunks (page_number=N)
Page N+1 文本 → parent chunks → child chunks (page_number=N+1)
```

### 问题场景

学术论文中一个完整段落可能跨页。例如第 4 页末尾：

> "我们提出了一种新的注意力机制，具体设计如下——"

第 5 页开头：

> "首先将 Q、K、V 分别投影到三个低秩空间中..."

模型检索到第 4 页末尾的 chunk 时，只能看到半句话，看不到下一页的方法描述。类似地，检索到第 5 页开头时缺少上文铺垫。

虽然父块有 200 字符重叠窗口，但物理页面边界可能让段落尾部和开头分开几百字符，重叠未必能覆盖。

### 方案对比

#### 方案 A：接受现状（不动）

**做法**：维持当前按页切块逻辑，不做任何跨页连接。

**优点**：
- 零开发成本，零维护成本
- 页码证据绝对精确，chunk 沿袭严格的页边界
- 现有 200 字符重叠已覆盖大部分页边界内容
- 模型检索到不完整信息时可主动补充检索

**缺点**：
- 跨页段落检索时可能丢失上下文，模型只能看到段落的半截
- 模型可能基于不完整信息给出不完整回答
- 补充检索消耗额外工具预算和时延
- 用户无法感知信息缺失，回答可信度受损

**适用**：跨页段落因拆分而断裂的概率低且影响有限的场景。

---

#### 方案 B：相邻页链接（推荐）

**做法**：构建 chunk 时给每页的首尾 parent 的 child 记录前后页的 chunk_id。检索命中边界 chunk 时自动拉取相邻页内容。

**具体设计**：

1. **写入阶段**：`_build_child_docs_from_pages()` 中，填充以下 metadata 字段：
   - `chev_chunk_ids`：指向下一页首 parent 中所有 child 的 chunk_id 列表（当前页最后一个 parent 的 child 才有）
   - `prev_chunk_ids`：指向前一页尾 parent 中所有 child 的 chunk_id 列表（当前页第一个 parent 的 child 才有）

2. **检索阶段**：`format_pdf_parent_record()` 或 `search_pdf` 末尾，对命中结果进行检查：
   - 如果 parent_record 的某个 child 含有 `next_chunk_ids`，额外从 Qdrant 按 chunk_id 拉取下一页内容
   - 如果含有 `prev_chunk_ids`，拉取前一页内容
   - 拉取到的相邻内容附加在检索结果末尾，标注 `（相邻页补充内容，页码=N+1）`

3. **检索增强逻辑**：
   - 默认只拉相邻 1 页，避免无限递归
   - 拉到的相邻 chunk 以短片段形式附在结果后方，不做新的 rerank
   - 若相邻 chunk 拉取失败（Qdrant 查询异常），静默跳过而非报错

**优点**：
- 页码证据仍然绝对精确（chunk 归属页不变，相邻内容是显式标注的附加上下文）
- 跨页段落可以自然衔接，模型看到完整语义
- 只在检索命中边界 chunk 时触发额外查询，非边界 chunk 零开销
- 相邻内容以标注形式出现（`相邻页补充内容，页码=N+1`），模型能区分主内容和附加上下文
- 实现复杂度适中，仅在写入和检索两处做增量改动

**缺点**：
- 写入时多一个 O(1) 的字典查找（用 `next_page_first_parent_id` 做 key）
- 检索时可能多 1-2 次 Qdrant 查询（仅在命中边界 chunk 时）
- 需要维护 chunk_id 的正确性（chunk_id 在构建时已生成，不涉及后续同步问题）
- 若 Qdrant 的 chunk_id payload index 缺失，按 id 查询会退化到 scroll（已建有 payload index 则不影响）

**选择理由**：
- 方案 A 的问题虽然是低频场景，但一旦发生，模型给出的不完整回答会降低用户信任
- 方案 B 改动量小（仅 `document.py` 构建逻辑 + `tools.py` 检索格式化），风险可控
- 相邻页链接没有改变切块策略本身，只是让边界 chunk 多了引用能力

### 实现详情

**写入阶段** (`agent/document.py:_build_child_docs_from_pages`):
- 收集每页的 parent_id 有序列表 `page_parent_ids: dict[int, list[str]]`
- 构建完所有 child_docs 后遍历补齐链接：
  - 某页最后一个 parent 的 children → `next_parent_id` = 下一页第一个 parent 的 id
  - 某页第一个 parent 的 children → `prev_parent_id` = 前一页最后一个 parent 的 id

**检索阶段** (`agent/tools.py:search_pdf`):
- `parent_hits` 积累时新增 `next_parent_id` / `prev_parent_id` 字段
- `_fetch_parent_text_by_id()` 按 parent_id 从 Qdrant 拉取父块文本（利用 `parent_id` payload index，scroll limit=1）
- 对 rerank 和 RRF 降级两条返回路径，均检查链接字段，命中时拉取并截取前 600 字符附在结果末尾

**拉取失败处理**：Qdrant 查询异常时静默跳过，不中断主检索链路。

### 状态

已实现。文件: `agent/document.py`, `agent/tools.py`。

---

## 痛点 2：默认入库路径静默丢弃图表内容

### 背景

当前 `process_pdf_pages()` 默认 `include_visual_descriptions=False`。在此路径下，文本页只走 `page.get_text()`，嵌入 PDF 的图表（实验结果图、模型结构图、对比表格）不会被提取，直接丢失。

### 问题场景

用户上传一篇论文，未勾选"表格/图表增强"开关。论文章 3 的实验结果页包含：
- 一段文字分析
- 一张准确率对比柱状图
- 一张消融实验表格

入库后只有文字被索引，柱状图和表格内容不可检索。用户问"这篇论文的消融实验结果如何？"——模型搜不到表格数据，可能基于文字分析给出不完整回答，甚至回退到说"论文中没有提供消融实验"。

### 设计原则

**学术论文场景下，图表与文字同等关键，不能静默丢弃任何内容。** 如果某个路径会丢数据，至少应该让用户和模型知道丢了什么。

### 讨论

当前方案如何演进：

- **短期**：默认路径下，对检测到有图表的文本页生成占位标记（如 `[本页包含未处理的图表，可重新上传并开启视觉增强以索引图表内容]`），让用户和模型知道缺失了什么
- **长期**：将视觉增强从 opt-in 改为默认开启，PP-Structure 表格提取（免费、本地）作为 baseline，GLM-4V 图表描述作为可选增强

### 状态

待讨论和实现。
