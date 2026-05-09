# main.py
# 当前项目入口。

from agent import AgentLoop

if __name__ == "__main__":
    from ui.gradio_app import create_gradio_ui

    demo = create_gradio_ui(app_factory=AgentLoop)
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
