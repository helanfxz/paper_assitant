# memory/compression.py
# 实现短期记忆的滑动窗口压缩机制。
# 当 session 内消息数超过 WINDOW_SIZE 时，取最早的 COMPRESS_BATCH 条对话消息，
# 用 fast_llm 生成摘要并全量写入 Qdrant 长期记忆，然后直接删除这批消息。
# 不再插入摘要 SystemMessage，历史信息通过每轮检索注入保证不丢失。

from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from memory.store import save_fact

WINDOW_SIZE = 20    # 消息条数上限（约 10 轮对话）
COMPRESS_BATCH = 4  # 每次压缩最早的 4 条（约 2 轮）


def compress_window(app, config: dict, memory_store, user_id: str, session_id: str, fast_llm):
    """
    在 ask() 开始时调用。
    若当前消息数超过 WINDOW_SIZE，压缩最早的 COMPRESS_BATCH 条对话消息（Human/AI），
    生成摘要全量写入 Qdrant，然后直接删除这批消息。
    长期记忆通过每轮 ask() 开始时的语义检索注入，保证历史信息不丢失。
    """
    state = app.get_state(config)
    messages = state.values.get("messages", [])
    if len(messages) <= WINDOW_SIZE:
        return

    conv_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    if len(conv_msgs) < COMPRESS_BATCH:
        return

    old_msgs = conv_msgs[:COMPRESS_BATCH]

    dialogue = "\n".join([
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:300]}"
        for m in old_msgs
        if m.content
    ])

    prompt = (
        f"请用一句话概括以下对话的核心内容，只输出摘要，不要其他内容：\n\n"
        f"{dialogue}"
    )

    try:
        summary = fast_llm.invoke(prompt).content.strip()
        if not summary:
            summary = dialogue[:100]
    except Exception:
        summary = dialogue[:100]

    save_fact(memory_store, summary, user_id, session_id, fact_type="auto_fact")
    print(f"[Memory] 压缩写入长期记忆: {summary[:60]}")

    removes = [RemoveMessage(id=m.id) for m in old_msgs]
    app.update_state(config, {"messages": removes})
    print(f"[Memory] 窗口压缩：{len(messages)} → {len(messages) - len(old_msgs)} 条消息")
