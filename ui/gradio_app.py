"""Gradio 前端层。"""

from __future__ import annotations

import queue
import threading
from typing import Any

import gradio as gr

from agent.session import delete_session, list_sessions


def create_gradio_ui(app_factory):
    """创建 Gradio 界面。"""
    state = {"app": None, "user_id": "pdf_user1"}

    def _session_choices(user_id: str):
        sessions = list_sessions(user_id)
        return [(f"{session['title']}  ({session['created_at'][:16]})", session["session_id"]) for session in sessions]

    def _note_choices():
        if state["app"] is None:
            return []
        choices = []
        for note in state["app"].list_notes():
            title = note["title"] or note["content"][:20]
            updated_at = note["updated_at"][:16] if note["updated_at"] else ""
            choices.append((f"{title}  ({updated_at})", note["note_id"]))
        return choices

    def _pending_view():
        if state["app"] is None:
            return "", gr.update(visible=False)
        summary = state["app"].get_pending_action_summary()
        current_action = state["app"].get_current_pending_action()
        return summary, gr.update(visible=current_action is not None)

    def new_session(user_id: str):
        if not user_id.strip():
            user_id = "pdf_user1"
        state["user_id"] = user_id
        state["app"] = app_factory(user_id=user_id)
        pending_summary, pending_area_state = _pending_view()
        return (
            gr.update(choices=_session_choices(user_id), value=state["app"].session_id),
            [],
            f"当前会话：{state['app'].session_id}",
            pending_summary,
            pending_area_state,
            "",
            gr.update(choices=_note_choices(), value=None),
            "",
            "",
            "",
            "笔记列表已刷新。",
        )

    def load_session(session_id: str):
        if not session_id:
            return [], "请先选择会话。", "", gr.update(visible=False), "", gr.update(choices=[], value=None), "", "", "", "笔记列表已刷新。"
        state["app"] = app_factory(user_id=state["user_id"], session_id=session_id)
        pending_summary, pending_area_state = _pending_view()
        return (
            state["app"].get_chat_history(),
            f"当前会话：{session_id}",
            pending_summary,
            pending_area_state,
            "",
            gr.update(choices=_note_choices(), value=None),
            "",
            "",
            "",
            "笔记列表已刷新。",
        )

    def refresh_list(user_id: str):
        if user_id.strip():
            state["user_id"] = user_id
        return gr.update(choices=_session_choices(state["user_id"]))

    def remove_session(session_id: str):
        if not session_id:
            return (
                gr.update(choices=_session_choices(state["user_id"]), value=None),
                [],
                "请先选择一个会话。",
                "",
                gr.update(visible=False),
                "",
                gr.update(choices=[], value=None),
                "",
                "",
                "",
                "",
            )
        delete_session(session_id)
        if state["app"] is not None and state["app"].session_id == session_id:
            state["app"] = None
        return (
            gr.update(choices=_session_choices(state["user_id"]), value=None),
            [],
            "请新建或选择一个会话。",
            "",
            gr.update(visible=False),
            "",
            gr.update(choices=[], value=None),
            "",
            "",
            "",
            "",
        )

    def load_pdf(pdf_file):
        if state["app"] is None:
            return "请先新建或选择会话。"
        if pdf_file is None:
            return "请先上传 PDF 文件。"
        result = state["app"].load_document(pdf_file)
        return f"成功：{result['message']}" if result["success"] else f"失败：{result['message']}"

    def chat(message: str, history: list[list[str]] | None):
        """流式聊天生成器。"""
        if state["app"] is None:
            current_history = list(history or [])
            current_history.append([message, "请先新建或选择会话。"])
            yield "", current_history, "", gr.update(visible=False), ""
            return

        if not message.strip():
            pending_summary, pending_area_state = _pending_view()
            yield "", history or [], pending_summary, pending_area_state, ""
            return

        current_history = list(history or [])
        current_history.append([message, ""])
        yield "", current_history, "", gr.update(visible=False), ""

        event_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        final_result = {"answer": "", "error": ""}

        def on_content_delta(delta: str) -> None:
            event_queue.put(("delta", delta))

        def run_ask() -> None:
            print("[UI] run_ask start")
            try:
                answer = state["app"].ask(message, on_content_delta=on_content_delta)
                print("[UI] run_ask success")
                final_result["answer"] = answer
                event_queue.put(("final", answer))
            except Exception as exc:
                print(f"[UI] run_ask error: {exc}")
                final_result["error"] = str(exc)
                event_queue.put(("error", f"请求执行失败：{exc}"))
            finally:
                print("[UI] run_ask done")
                event_queue.put(("done", ""))

        worker = threading.Thread(target=run_ask, daemon=True)
        worker.start()

        streamed_answer = ""
        is_done = False
        while not is_done:
            try:
                event_type, payload = event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if event_type == "delta":
                streamed_answer += payload
                current_history[-1][1] = streamed_answer
                yield "", current_history, "", gr.update(visible=False), ""
                continue

            if event_type == "final":
                current_history[-1][1] = payload
                yield "", current_history, "", gr.update(visible=False), ""
                continue

            if event_type == "error":
                current_history[-1][1] = payload
                yield "", current_history, "", gr.update(visible=False), ""
                continue

            if event_type == "done":
                is_done = True

        pending_summary, pending_area_state = _pending_view()
        if final_result["answer"]:
            current_history[-1][1] = final_result["answer"]
        elif final_result["error"]:
            current_history[-1][1] = final_result["error"]
        yield "", current_history, pending_summary, pending_area_state, ""

    def _run_pending_action(history: list[list[str]] | None, decision: str, feedback: str = ""):
        """处理确认型动作，不新增可见用户气泡。"""
        if state["app"] is None:
            yield history or [], "", gr.update(visible=False), "", "请先新建或选择会话。"
            return

        current_history = list(history or [])
        pending_summary, pending_area_state = _pending_view()
        yield current_history, pending_summary, pending_area_state, feedback, "正在处理确认操作..."

        try:
            state["app"].continue_pending_action(decision=decision, feedback=feedback)
            note = "确认操作已处理。"
        except Exception as exc:
            note = f"确认操作处理失败：{exc}"

        pending_summary, pending_area_state = _pending_view()
        yield state["app"].get_chat_history(), pending_summary, pending_area_state, "", note

    def approve_pending(history: list[list[str]] | None):
        yield from _run_pending_action(history, decision="approve")

    def reject_pending(history: list[list[str]] | None):
        yield from _run_pending_action(history, decision="reject")

    def feedback_pending(history: list[list[str]] | None, feedback_text: str):
        cleaned_feedback = feedback_text.strip()
        if not cleaned_feedback:
            pending_summary, pending_area_state = _pending_view()
            yield history or [], pending_summary, pending_area_state, feedback_text, "请先填写补充意见。"
            return
        yield from _run_pending_action(history, decision="feedback", feedback=cleaned_feedback)

    def refresh_notes():
        return gr.update(choices=_note_choices(), value=None), "", "", "", "笔记列表已刷新。"

    def load_note(note_id: str):
        if state["app"] is None:
            return "", "", "", "请先新建或选择会话。"
        if not note_id:
            return "", "", "", ""
        note = state["app"].get_note(note_id)
        if note is None:
            return "", "", "", "未找到对应笔记。"
        note_title = note["title"]
        note_label = note_title or note["note_id"]
        return note["note_id"], note_title, note["content"], f"已加载笔记：{note_label}"

    def save_note(note_id: str, title: str, content: str):
        if state["app"] is None:
            return gr.update(), "", "", "", "请先新建或选择会话。"
        save_message = state["app"].save_note(content=content, title=title, note_id=note_id)
        note_choices = _note_choices()
        active_note_id = note_id or (note_choices[0][1] if note_choices else "")
        return (
            gr.update(choices=note_choices, value=active_note_id or None),
            active_note_id,
            title,
            content,
            save_message,
        )

    def delete_note(note_id: str):
        if state["app"] is None:
            return gr.update(), "", "", "", "请先新建或选择会话。"
        if not note_id:
            return gr.update(), "", "", "", "请先选择要删除的笔记。"
        delete_message = state["app"].delete_note(note_id)
        return gr.update(choices=_note_choices(), value=None), "", "", "", delete_message

    with gr.Blocks(title="论文阅读助手", theme=gr.themes.Soft()) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 论文阅读助手")
                user_id_input = gr.Textbox(
                    label="用户ID",
                    value="pdf_user1",
                    container=False,
                    placeholder="输入用户ID",
                )
                new_btn = gr.Button("新建对话", variant="primary", size="sm")
                refresh_btn = gr.Button("刷新列表", size="sm")
                delete_session_btn = gr.Button("删除会话", size="sm")

                session_radio = gr.Radio(
                    label="历史会话",
                    choices=_session_choices("pdf_user1"),
                    interactive=True,
                )

                with gr.Accordion("上传 PDF 文档", open=False):
                    pdf_upload = gr.File(
                        label="选择文件",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    load_btn = gr.Button("加载文档", size="sm")
                    load_output = gr.Textbox(label="状态", interactive=False, lines=2)
                    load_btn.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])

                with gr.Accordion("研究笔记", open=False):
                    note_picker = gr.Dropdown(label="已有笔记", choices=[], value=None, interactive=True)
                    note_id_box = gr.Textbox(value="", visible=False)
                    note_title = gr.Textbox(label="笔记标题", placeholder="可选")
                    note_content = gr.Textbox(
                        label="笔记内容",
                        lines=8,
                        placeholder="输入研究结论、论文比较结论或阅读笔记",
                    )
                    with gr.Row():
                        refresh_notes_btn = gr.Button("刷新", size="sm")
                        save_note_btn = gr.Button("保存/更新", variant="primary", size="sm")
                        delete_note_btn = gr.Button("删除", size="sm")
                    note_status = gr.Textbox(label="笔记状态", interactive=False, lines=2)

            with gr.Column(scale=4):
                session_label = gr.Textbox(
                    value="请新建或选择一个会话。",
                    interactive=False,
                    container=False,
                    show_label=False,
                )
                chatbot = gr.Chatbot(label="", height=520, show_label=False)

                with gr.Group(visible=False) as pending_area:
                    pending_summary = gr.Textbox(
                        label="待确认操作",
                        interactive=False,
                        lines=3,
                    )
                    with gr.Row():
                        approve_btn = gr.Button("同意", variant="primary", size="sm")
                        reject_btn = gr.Button("拒绝", size="sm")
                    feedback_box = gr.Textbox(
                        label="补充意见",
                        placeholder="如果需要修正保存内容，在这里填写补充意见",
                        lines=3,
                    )
                    feedback_btn = gr.Button("提交补充意见", size="sm")
                    pending_status = gr.Textbox(label="处理状态", interactive=False, lines=2)

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入消息，按 Enter 发送...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        new_btn.click(
            new_session,
            inputs=[user_id_input],
            outputs=[session_radio, chatbot, session_label, pending_summary, pending_area, pending_status, note_picker, note_id_box, note_title, note_content, note_status],
        )
        refresh_btn.click(refresh_list, inputs=[user_id_input], outputs=[session_radio])
        delete_session_btn.click(
            remove_session,
            inputs=[session_radio],
            outputs=[session_radio, chatbot, session_label, pending_summary, pending_area, pending_status, note_picker, note_id_box, note_title, note_content, note_status],
        )
        session_radio.change(
            load_session,
            inputs=[session_radio],
            outputs=[chatbot, session_label, pending_summary, pending_area, pending_status, note_picker, note_id_box, note_title, note_content, note_status],
        )
        msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot, pending_summary, pending_area, pending_status])
        send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot, pending_summary, pending_area, pending_status])
        approve_btn.click(
            approve_pending,
            inputs=[chatbot],
            outputs=[chatbot, pending_summary, pending_area, feedback_box, pending_status],
        )
        reject_btn.click(
            reject_pending,
            inputs=[chatbot],
            outputs=[chatbot, pending_summary, pending_area, feedback_box, pending_status],
        )
        feedback_btn.click(
            feedback_pending,
            inputs=[chatbot, feedback_box],
            outputs=[chatbot, pending_summary, pending_area, feedback_box, pending_status],
        )

        refresh_notes_btn.click(
            refresh_notes,
            outputs=[note_picker, note_id_box, note_title, note_content, note_status],
        )
        note_picker.change(
            load_note,
            inputs=[note_picker],
            outputs=[note_id_box, note_title, note_content, note_status],
        )
        save_note_btn.click(
            save_note,
            inputs=[note_id_box, note_title, note_content],
            outputs=[note_picker, note_id_box, note_title, note_content, note_status],
        )
        delete_note_btn.click(
            delete_note,
            inputs=[note_id_box],
            outputs=[note_picker, note_id_box, note_title, note_content, note_status],
        )

    return demo
