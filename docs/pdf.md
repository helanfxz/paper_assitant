# PDF 解析链路

从 PDF 上传到可检索的完整链路拆解。代码变更时需同步更新本文档。

---

## 第 1 块：上传入口与任务调度

**入口**：用户通过 Gradio UI 上传 PDF → `load_document()`（`agent/document.py`）

**流程**：
```
上传 → 计算 file_hash（SHA256）
  → 检查是否已入库（同 hash + 同 ingestion_profile → 复用，跳过所有后续步骤）
  → 若存在同名旧文档：清理旧 Qdrant points + 旧 BM25 记录
  → process_pdf_pages() 获取 PageContent[]
  → _build_child_docs_from_pages() 做父子块切分
  → pdf_store.add_documents() 写入 Qdrant
  → lexical_index.index_documents() 写入 BM25
  → LLM 提取标题和摘要 → register_document() 写入 documents.json
```

**技术实现**：
- `hashlib.sha256`：文件指纹，分块读取（1MB）避免大文件内存峰值
- `DocumentIngestionJobRegistry`（`document.py:29-133`）：进程内后台任务表，线程池执行，UI 通过 `job_id` 轮询状态
- `_is_same_registered_document()`：按 filename + file_hash + ingestion_profile 三要素判断是否可复用
- `_replace_existing_pdf_records()`：同名重入库前清理旧 Qdrant points + 旧 BM25 记录，避免检索混入旧内容

**设计要点**：
- 后台异步入库，上传不阻塞 UI
- 同文件 + 同配置重复上传自动复用
- 内容或配置变更时先清理旧数据再写入

---

## 第 2 块：页面类型分类

**入口**：`classify_pages()`（`agent/scanned_pdf_processor.py:275`）

**策略**：偏保守——只有几乎确定是扫描页时才走 OCR，其余默认文本路径。

**判断规则**（`classify_page_metrics:230-244`）：
```
有效字符数 >= SCANNED_PAGE_CHAR_THRESHOLD（默认 100） → text
text_block_count == 0 且 图片覆盖面积 >= 60%         → scanned
字符数 == 0 且 图片覆盖面积 >= 35%                    → scanned
其他所有情况                                          → text
```

**三项检测指标**：

| 指标 | 获取方式 | 作用 |
|---|---|---|
| 有效字符数 | `page.get_text()` 去除换行和空格后计数 | 主要 gate，高字符数直接判定为文本页 |
| text block 数量 | `page.get_text("blocks")`，排除图片 block（`block[6] != 0`） | 区分"扫描页无文字"和"排版稀疏但仍有文字" |
| 图片覆盖面积 | `page.get_images()` 遍历 xref + `page.get_image_rects()` 累加面积 | 辅助判断，识别全页图片型扫描页 |

**已知局限**：
- 图片覆盖面积为估算值（同一张图在 PDF 内部可能有多个 xref 对象导致重复累加），但上限 `min(ratio, 1.0)` 兜底，且分类不以覆盖面积为主要 gate，实际不影响分类结果
- 未做 OCR 抽样验证（计划中提到但未实现），当前多因素判断已满足准确率要求

---

## 第 3 块：页面处理编排器

**入口**：`process_pdf_pages()`（`agent/scanned_pdf_processor.py:414`）

**流程**：
```
classify_pages() 获取分类结果 → 打开 PDF 文档
  → 逐页循环：
      ├─ 文本页：page.get_text().strip()
      │   → PageContent.from_native_text()
      │
      ├─ 扫描页：render_page_image() → ocr.extract_text()
      │   → PageContent.from_ocr()
      │
      └─ 若开启视觉增强（include_visual_descriptions=True）：
          render_page_image() → vision_analyzer.analyze_page()
          → layout_result_to_page_contents()
          → 生成独立的 PageContent（table_extraction / vision_model）
  → 写入缓存 → 返回 PageContent[]
```

**逐页串行策略**：
- OCR 内部仍是 for 循环逐张处理，不存在批量推理优势
- 内存占用低（只保留当前页渲染图片）
- 单页失败不中断后续页面处理

**技术实现**：
- `render_page_image()`：`page.get_pixmap()` 直接渲染为 PIL Image，不依赖 poppler，减少外部依赖
- 扫描页渲染的 `page_image` 可复用给后续视觉增强步骤，避免同一页渲染两次

**缓存策略**（`_page_cache_path:181-200`）：
```
cache_key = SHA256(
    PDF_PAGE_CACHE_VERSION +
    file_sha256 +
    ocr_profile（引擎类型 + 语言 + DPI + 阈值）+
    include_visual_descriptions +
    vision_profile（分析器类型 + vision_model 名称 + token 限制）
)
```
- 缓存路径：`PDF_PAGE_CACHE_DIR / {cache_key}.json`
- 原子写入：先写 `.tmp` 文件，再 `replace` 到目标路径，防止写入中断损坏缓存
- 配置变更（OCR 语言、模型切换、DPI 调整）产生新 cache key，自动触发重新处理

---

## 第 4 块：OCR 引擎

**入口**：`OcrEngine.extract_text()`（`agent/ocr_engine.py:98`）

**策略**：PaddleOCR 为主引擎，Tesseract 为降级备选。

**降级链**：
```
PaddleOCR 已安装 → 尝试调用 → 成功 → 返回（engine="paddleocr"，逐行平均置信度）
                             → 失败 → _paddle_available = False
                                    → Tesseract（engine="tesseract"，固定 confidence=0.5）
PaddleOCR 未安装 → Tesseract
两者都不可用 → 返回错误占位文本（engine="failed"）
```

**两种 OCR 引擎对比**：

| | PaddleOCR | Tesseract |
|---|---|---|
| 来源 | 百度开源深度学习 OCR | Google 开源传统 OCR |
| 原理 | 神经网络检测 + 识别两阶段 | 基于字符模式匹配 |
| 中文准确率 | 高，专门训练中文模型 | 低，中文非其强项 |
| 安装方式 | `pip install paddleocr` | `apt install tesseract-ocr` |
| 资源消耗 | 大（PaddlePaddle 框架 + 模型文件） | 小（系统包管理器） |

**设计要点**：
- 延迟初始化：`_init_paddle()` 首次调用时才 import，避免启动时内存峰值
- 永久降级：一次 PaddleOCR 失败后标记不可用，不再重试。失败通常是环境问题（缺依赖、显存不足），反复重试无意义
- 单页失败不中断：异常返回占位文本 `(OCR 失败: ...)`，编排器继续处理后续页

---

## 第 5 块：PageContent 中间结构与证据标注

**数据结构**：`PageContent` dataclass（`agent/scanned_pdf_processor.py:64-119`）

```
PageContent:
  page_number      → 页码
  page_type        → "text" | "scanned" | "mixed"
  text             → 提取的文本内容
  content_source   → "native_pdf_text" | "ocr" | "vision_model" | "table_extraction"
  source_filename  → 来源 PDF 文件名
  ocr_engine       → "paddleocr" | "tesseract" | "none"
  confidence       → 置信度（原生文本 = 1.0，OCR = 引擎返回，视觉 = 模型返回）
  generated        → 是否为 AI 模型生成的描述（图表描述 = True）
  has_figure       → 是否包含图表区域
  has_table        → 是否包含表格区域
```

**四种内容来源及其可靠性**：

| content_source | 含义 | 能否作为论文原文 | 可靠性 |
|---|---|---|---|
| `native_pdf_text` | PDF 原生文本层提取 | 是 | 最高 |
| `ocr` | 扫描页 OCR 识别 | 是，但标注 OCR 引擎和置信度 | 高（PaddleOCR）/ 中（Tesseract）|
| `vision_model` | GLM-4V 对图表的自然语言描述 | **否**，由 AI 生成 | 低，不能当原文引用 |
| `table_extraction` | PP-Structure 提取的结构化表格 | 可作为表格数据参考 | 中高 |

**设计原则**：内容和来源绑定在同一结构中。chunk metadata 完整携带所有证据字段，`search_pdf` 和 system prompt 共同保障模型区分"原文"和"模型生成内容"。

**Markdown 拼接**：`pages_to_markdown()` 按页拼接带 HTML 注释锚点 `<!-- page: N; page_type: ... -->` 的 Markdown。此拼接结果**仅用于 LLM 元数据提取**（标题和摘要），检索索引走按页切块路径，与此无关。

---

## 第 6 块：父子块切分

**入口**：`_build_child_docs_from_pages()`（`agent/document.py:336`）

**策略**：每页独立做 parent/child split，确保 chunk 稳定继承 `page_number`。切分后补齐跨页链接。

**切分参数**：

| 参数 | 值 | 用途 |
|---|---|---|
| parent chunk_size | 1500 | 返回给模型的上下文单位 |
| parent chunk_overlap | 200 | 父块间冗余，缓和页边界断裂 |
| child chunk_size | 400 | 向量检索的精细匹配单位 |
| child chunk_overlap | 50 | 子块间冗余 |

**流程**：
```
Page N 文本 → parent_splitter → [parent_1, parent_2, ...]
  → 每个 parent → child_splitter → [child_1, child_2, ...]
  → 每个 child 继承：source, page_number, page_type, content_source, confidence, ...
  → 收集 page_parent_ids: {page_N: [p_id_1, p_id_2, ...]}
  → 补齐跨页链接：
      - 页 N 最后一个 parent 的所有 child → next_parent_id = 页 N+1 第一个 parent 的 id
      - 页 N 第一个 parent 的所有 child → prev_parent_id = 页 N-1 最后一个 parent 的 id
```

**跨页链接机制**（详见 `docs/question_solve.md` 痛点 1）：
- 写入阶段：边界 child 的 metadata 携带 `next_parent_id` / `prev_parent_id`
- 检索阶段：`_fetch_parent_text_by_id()` 利用 Qdrant `parent_id` payload index 按需拉取相邻页
- 拉取失败静默跳过，不中断主检索

---

## 第 7 块：向量库 + BM25 写入

**入口**：`load_document()` → `pdf_store.add_documents()` + `lexical_index.index_documents()`

**Qdrant（向量库）**：
- 集合名：`pdf_knowledge`
- 向量模型：智谱 Embedding-3
- payload index 字段（`agent/memory.py:30-38`）：`user_id`, `source`, `parent_id`, `chunk_id`, `page_type`, `content_source`, `evidence_type`
- 共享范围：`PDF_SHARED_SCOPE`，所有用户共享同一知识库

**BM25（词法索引）**：
- 存储：SQLite（`lexical_index.py`）
- 分词器：`go-lemmekit`（Go 实现的词形还原）
- 共享范围：与 Qdrant 一致，`PDF_SHARED_SCOPE`
- 回补机制：`ensure_source_indexed_from_vector_store()` 从 Qdrant 回补旧文档到 BM25，保证新老数据检索行为一致

---

## 第 8 块：检索（search_pdf）

**入口**：`search_pdf()`（`agent/tools.py:287`）

**完整检索链路**：
```
用户 query
  → MQE 查询扩展（fast_llm 生成 1-3 条变体，覆盖不同检索角度）
  → 多路并发检索（ThreadPoolExecutor）：
      ├─ 向量检索（每条 query → Qdrant similarity_search，top_k=3）
      └─ BM25 检索（每条 query → lexical_index.search，top_k=3）
  → RRF 融合（k=60，多路 child chunk 按排名加权求和）
  → parent 聚合（同 parent_id 的 child 得分累加）
  → top-8 parent 进入 rerank（qwen3-rerank）
  → 跨页上下文补充（检测 next_parent_id / prev_parent_id → 拉取相邻页 parent_text）
  → 返回 top-3 parent + 证据 metadata
```

**各环节关键参数**：

| 参数 | 值 | 说明 |
|---|---|---|
| MQE_EXPAND_COUNT | 3 | 最多生成 3 条扩展查询 |
| MQE_MIN_QUERY_LEN | 4 | 过短查询不做扩展以避免无意义短语 |
| VECTOR_TOP_K | 3 | 每条查询向量检索候选数 |
| BM25_TOP_K | 3 | 每条查询词法检索候选数 |
| RRF_K | 60 | Reciprocal Rank Fusion 平滑参数 |
| RERANK_PARENT_LIMIT | 8 | 进入 rerank 的父块上限 |

**降级路径**：
- BM25 不可用 → 退化为纯向量检索
- rerank 失败 → 使用 RRF 融合结果直接返回
- 跨页上下文拉取失败 → 静默跳过，不影响主结果

**检索结果格式**（`format_pdf_parent_record()`，`tools.py:48-64`）：
```
来源文档：xxx.pdf
页码：12
页面类型：text / scanned / mixed
来源类型：native_pdf_text / ocr / vision_model / table_extraction
自动生成内容：是 / 否
父块文本...
（前一页补充内容）：...
（下一页补充内容）：...
```

---

## API 与依赖总览

| 环节 | 技术 | 模型 / 引擎 |
|---|---|---|
| 文本提取 | PyMuPDF (fitz) | — |
| OCR | PaddleOCR / Tesseract | PaddleOCR ch 模型 |
| 表格提取 | PP-Structure | — |
| 图表描述 | 智谱 API | `glm-4v-flash` |
| 文件指纹 | hashlib.sha256 | — |
| 文本切分 | LangChain RecursiveCharacterTextSplitter | — |
| 向量化 | 智谱 API | `Embedding-3` |
| BM25 | go-lemmekit + SQLite | — |
| MQE 查询扩展 | DeepSeek API | `deepseek-chat` |
| Rerank | 阿里百炼 API | `qwen3-rerank` |
| 元数据抽取 | DeepSeek API | `deepseek-chat` |
