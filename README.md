# 学术论文阅读助手

一个面向研究生和科研人员的论文阅读助手，当前采用“单循环 harness 应用 + 程序层控制”的实现方式，重点解决：

- 论文检索与问答
- 长对话续航
- 跨 session 用户偏好记忆
- 学习笔记沉淀

## 当前架构

项目已经从早期的 LangGraph 多节点实现，收敛为更轻量的单循环架构：

- `main.py`
  薄入口，只负责启动应用和兼容旧导出
- `agent/app.py`
  负责主循环编排、错误恢复接入、偏好检测与确认、会话消息提交
- `agent/memory.py`
  负责 Qdrant 记忆、滑动窗口压缩、用户偏好状态管理；其中 `UserPreferenceProfileStore` 管理跨 session 偏好
- `tools.py`
  负责运行时工具构建；`build_runtime_tools` 会按当前 session 注入文档检索、记忆回忆、笔记管理和统计工具
- `agent/document.py`
  负责 PDF 解析、父子块切分、文档元数据注册，以及把子块同步写入向量库和持久化词法索引
- `lexical_index.py`
  负责 SQLite 持久化词法索引，为 BM25 检索提供独立倒排索引
- `session.py`
  负责 session 元数据和消息历史持久化
- `ui/gradio_app.py`
  负责 Gradio 交互界面

## 三层记忆

### 1. Profile Memory

用户长期偏好，跨 session 持久化。

- 不做检索
- 每轮全量注入 system prompt
- 采用 `scope + type` 槽位更新
- 同槽位新值覆盖旧值

当前支持的偏好类型：

- `language`
- `format`
- `detail_level`
- `focus`
- `avoid`

当前支持的作用域：

- `global`
- `paper_summary`
- `paper_compare`

### 2. Study Notes

跨 session 的研究笔记，存放在独立的 `study_notes` collection 中。

- 用户主动保存
- 支持单独 CRUD
- 按问题语义检索
- 有分数阈值过滤
- 用于补充当前问题相关背景

### 3. Session Memory

长对话超过窗口后，会把旧消息压缩成摘要写入会话记忆库。

- 最近消息保留在消息窗口中
- 更早消息压缩后保存在 Qdrant
- 作用域仅限当前 session

## 用户偏好写入流程

当前实现采用“规则筛选 + fast llm 分类 + 程序层确认”的 hybrid 方案：

1. 先用规则判断当前用户输入是否像长期偏好候选
2. 只有命中候选时才调用 `fast_llm`
3. `fast_llm` 只输出 JSON
4. 程序用 `Pydantic` 验证 JSON 结构
5. 根据置信度决定：
   - `high`：自动保存
   - `medium`：程序层发起确认
   - `low`：不保存
6. 删除、清空等敏感操作强制确认

说明：

- Profile 不是“碎片化记忆列表”，而是“当前生效的偏好状态”
- 中置信度确认由程序层控制，不依赖模型自行记住

## 文档检索流程

文档检索仍保留论文阅读助手的核心能力：

1. PDF 转 Markdown
2. 父块 / 子块切分
3. 子块同时写入 Qdrant 和持久化词法索引
4. 检索时做 MQE 扩展查询
5. 向量检索与 BM25 检索并行召回
6. 用 RRF 融合多路 child chunk 结果
7. 聚合父块
8. 使用独立 `rerank_llm` 做最终重排

## 依赖

当前核心依赖包括：

- `langchain_openai`
- `langchain_qdrant`
- `qdrant_client`
- `pymupdf4llm`
- `gradio`
- `pydantic`

## 启动方式

```bash
./venv/bin/python main.py
```

如果需要手动安装新增依赖：

```bash
./venv/bin/pip install pydantic
```
