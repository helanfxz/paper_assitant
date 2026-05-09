# 问题与解决记录

---

## 1. 消息压缩导致 API 报 400

**问题**

对话消息超过窗口上限时，系统会把最早的一批消息压缩成摘要并删除。但压缩时只删除了 `user` 和 `assistant` 消息，没有同步删除夹在中间的 `tool` 消息。结果是 `assistant(tool_calls)` 被删掉了，对应的 `tool` 消息留了下来，形成孤立的 tool 消息。下一轮发给模型时，API 报错：

> Messages with role 'tool' must be a response to a preceding message with 'tool_calls'

**原因**

API 要求 `tool` 消息前面必须有对应的 `assistant(tool_calls)` 消息，两者通过 `tool_call_id` 绑定，缺一不可。

**解决方法**

把消息列表按"轮次"分组，每条 `user` 消息开启一轮，该轮包含它之后直到下一条 `user` 消息之前的所有消息（含 `tool` 消息）。压缩时以整轮为单位删除，`assistant(tool_calls)` 和对应的 `tool` 消息永远作为整体一起移除，不会出现孤立 tool 消息。

---

## 2. 偏好保存工具因参数不合法而静默失败

**问题**

用户表达长期偏好后，模型调用 `save_preference` 工具时传入了 `type="response_style"`，但系统只允许 `language / format / detail_level / focus / avoid` 这五个值。`apply_preference_update` 内部用 `TYPE_LABELS[type]` 取标签时抛出 `KeyError`，异常被 `except` 吞掉，工具返回了"执行失败"字符串，模型随即回答"抱歉，系统未能自动保存"。整个过程没有任何控制台输出，难以排查。

**解决方法**

两层防护：
1. 在工具描述里明确列出允许的 `scope` 和 `type` 枚举值，让模型尽量传正确的参数。
2. 在 `apply_preference_update` 入口加校验，`scope` 或 `type` 不合法时直接返回明确的错误字符串，而不是抛异常。同时在 `_execute_pending_action` 的 `except` 里加打印，方便排查。

---

## 3. 确认操作后前端出现空气泡

**问题**

用户点"同意"确认偏好后，前端聊天框里出现了一个没有用户消息的空气泡，显示模型的确认回答。

**原因**

`continue_pending_action` 执行完后，runner 产生的 assistant 回答会通过 `_commit_turn_messages` 写入 session。此时 session 里出现了两条连续的 assistant 消息，中间没有 user 消息。`get_chat_history` 把历史转成 Gradio 的 `[[user, assistant]]` 格式时，第二条 assistant 没有对应的 user，被渲染成空气泡。

**解决方法**（待实现）

在 `_commit_turn_messages` 之前，把 session 里最后一条 user 消息的内容替换成确认操作的摘要（例如"用户同意保存偏好：global / format"），这样 assistant 的确认回答就有对应的 user 消息，不会出现孤立气泡。
