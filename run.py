import gradio as gr

# server も import に追加します
from python import download, convert, quantize, mmproj, server

with gr.Blocks(title="Simple-llama-studio") as main_ui:
    gr.Markdown("# Simple-llama-studio")
    gr.Markdown("タブを切り替えて、「ダウンロード」→「GGUF変換」→「量子化」→「mmproj抽出」→「サーバー起動」の作業を一貫して行えます。")
    
    with gr.Tab("1. ダウンロード"):
        download.demo.render()
        
    with gr.Tab("2. GGUF変換"):
        convert.demo.render()
        
    with gr.Tab("3. 量子化"):
        quantize.demo.render()
        
    with gr.Tab("4. mmproj抽出"):
        mmproj.demo.render()
        
    # ★5つ目のタブとして追加
    with gr.Tab("5. サーバー起動"):
        server.demo.render()

if __name__ == "__main__":
    main_ui.launch(server_name="0.0.0.0", inbrowser=True)
