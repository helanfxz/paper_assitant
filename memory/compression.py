# memory/compression.py
# 实现短期记忆的滑动窗口压缩机制。
# 当 session 内消息数超过 WINDOW_SIZE 时，取最早的 COMPRESS_BATCH 条消息，
# 用 fast_llm 生成摘要并打重要性评分；评分达到阈值则晋升为 Qdrant 长期记忆，
# 之后将这批消息替换为一条摘要消息写回 checkpointer，保持窗口大小可控。

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from memory.store import save_fact

WINDOW_SIZE = 20       # 消息条数上限（约 10 轮对话）
COMPRESS_BATCH = 4     # 每次压缩最早的 4 条（约 2 轮）
PROMOTE_THRESHOLD = 3  # 重要性评分 >= 3 则晋升长期记忆


def compress_window(app, config: dict, memory_store, user_id: str, session_id: str, fast_llm):
    """
    在 ask() 开始时调用。
    若当前消息数超过 WINDOW_SIZE，压缩最早的 COMPRESS_BATCH 条对话消息（Human/AI），
    重要的晋升 Qdrant，全部替换为一条摘要 SystemMessage 写回 checkpointer。
    跳过 SystemMessage（历史摘要），避免压缩内容为空或误删旧摘要。
    """
    state = app.get_state(config)
    messages = state.values.get("messages", [])
    if len(messages) <= WINDOW_SIZE:
        return

    # 只从 Human/AI 消息中取，跳过 SystemMessage
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
        f"请对以下对话做两件事，严格按格式输出：\n"
        f"摘要：<用一句话概括核心内容>\n"
        f"评分：<1到5的整数，5=含用户关键偏好/目标/结论，1=纯闲聊>\n\n"
        f"对话：\n{dialogue}"
    )

    try:
        raw = fast_llm.invoke(prompt).content
        summary_line = next((l for l in raw.splitlines() if "摘要" in l), "")
        score_line = next((l for l in raw.splitlines() if "评分" in l), "")
        summary = summary_line.split("：", 1)[-1].strip() if summary_line else dialogue[:100]
        score_digits = "".join(filter(str.isdigit, score_line))
        score = max(1, min(5, int(score_digits[0]))) if score_digits else 1
    except Exception:
        summary = dialogue[:100]
        score = 1

    if score >= PROMOTE_THRESHOLD:
        save_fact(memory_store, summary, user_id, session_id, fact_type="auto_fact")
        print(f"[Memory] 晋升长期记忆(score={score}): {summary[:60]}")

    # 用 RemoveMessage 删除旧消息，再插入摘要
    # 注意：add_messages reducer 对 update_state 是追加语义，必须显式删除
    removes = [RemoveMessage(id=m.id) for m in old_msgs]
    summary_msg = SystemMessage(content=f"【早期对话摘要】: {summary}")
    app.update_state(config, {"messages": removes + [summary_msg]})
    print(f"[Memory] 窗口压缩：{len(messages)} → {len(messages) - len(old_msgs) + 1} 条消息")
