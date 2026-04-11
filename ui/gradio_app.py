# ui/gradio_app.py
# Gradio 前端 UI 层，仿 ChatGPT 布局：
# 左侧栏：用户ID、新建对话、历史会话列表、上传PDF
# 右侧主区：对话历史 + 输入框
# 点击历史会话直接加载该会话的消息记录，无需额外操作。

import gradio as gr
from session.manager import list_sessions


def create_gradio_ui(app_factory):
    state = {"app": None, "user_id": "pdf_user1"}

    def _session_choices(user_id: str):
        sessions = list_sessions(user_id)
        return [(f"{s['title']}  ({s['created_at'][:16]})", s["session_id"]) for s in sessions]

    def new_session(user_id: str):
        if not user_id.strip():
            user_id = "pdf_user1"
        state["user_id"] = user_id
        state["app"] = app_factory(user_id=user_id)
        choices = _session_choices(user_id)
        return (
            gr.update(choices=choices, value=state["app"].session_id),
            [],   # 清空对话框
            f"当前会话：{state['app'].session_id}",
        )

    def load_session(session_id: str):
        if not session_id:
            return [], "请先选择会话"
        state["app"] = app_factory(user_id=state["user_id"], session_id=session_id)
        history = state["app"].get_chat_history()
        return history, f"当前会话：{session_id}"

    def refresh_list(user_id: str):
        if user_id.strip():
            state["user_id"] = user_id
        return gr.update(choices=_session_choices(state["user_id"]))

    def load_pdf(pdf_file):
        if state["app"] is None:
            return "❌ 请先新建或选择会话"
        if pdf_file is None:
            return "❌ 请上传 PDF 文件"
        result = state["app"].load_document(pdf_file)
        return f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"

    def chat(message: str, history: list):
        if state["app"] is None:
            return "", history + [[message, "❌ 请先新建或选择会话"]]
        if not message.strip():
            return "", history
        response = state["app"].ask(message)
        history.append([message, response])
        return "", history

    with gr.Blocks(title="Multi-Agent 文档问答助手", theme=gr.themes.Soft(),
                   css=".session-list .wrap {gap: 4px}") as demo:

        with gr.Row(equal_height=True):

            # ── 左侧栏 ──────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 文档问答助手")
                user_id_input = gr.Textbox(
                    label="用户ID", value="pdf_user1", container=False,
                    placeholder="输入用户ID"
                )
                new_btn = gr.Button("＋ 新建对话", variant="primary", size="sm")
                refresh_btn = gr.Button("↻ 刷新列表", size="sm")

                session_radio = gr.Radio(
                    label="历史会话",
                    choices=_session_choices("pdf_user1"),
                    interactive=True,
                    elem_classes="session-list",
                )

                with gr.Accordion("上传 PDF 文档", open=False):
                    pdf_upload = gr.File(
                        label="选择文件", file_types=[".pdf"], type="filepath"
                    )
                    load_btn = gr.Button("加载文档", size="sm")
                    load_output = gr.Textbox(label="状态", interactive=False, lines=2)
                    load_btn.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])

            # ── 右侧主区 ─────────────────────────────────────
            with gr.Column(scale=4):
                session_label = gr.Textbox(
                    value="请新建或选择一个会话", interactive=False,
                    container=False, show_label=False
                )
                chatbot = gr.Chatbot(label="", height=520, show_label=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入消息，按 Enter 发送...",
                        show_label=False, scale=5, container=False
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        # ── 事件绑定 ──────────────────────────────────────────
        new_btn.click(
            new_session,
            inputs=[user_id_input],
            outputs=[session_radio, chatbot, session_label],
        )
        refresh_btn.click(
            refresh_list,
            inputs=[user_id_input],
            outputs=[session_radio],
        )
        session_radio.change(
            load_session,
            inputs=[session_radio],
            outputs=[chatbot, session_label],
        )
        msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

    return demo
