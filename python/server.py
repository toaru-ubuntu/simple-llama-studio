import gradio as gr
import subprocess
import sys 
import os
from pathlib import Path
from collections import deque

# ==========================================
# ディレクトリ設定 (config.yamlから読み込み)
# ==========================================
from python.config import BASE_DIR, MODELS_DIR

QUANTIZE_DIR = MODELS_DIR / "quantize"
GGUF_DIR = MODELS_DIR / "gguf"
MMPROJ_DIR = MODELS_DIR / "mmproj"
# ==========================================

server_process = None
chat_process = None 

def get_inference_models():
    models = []
    
    if QUANTIZE_DIR.exists():
        for f in QUANTIZE_DIR.iterdir():
            if f.is_file() and f.suffix == ".gguf":
                models.append(f"quantize/{f.name}")
                
    if GGUF_DIR.exists():
        for f in GGUF_DIR.iterdir():
            if f.is_file() and f.suffix == ".gguf":
                models.append(f"gguf/{f.name}")
                
    return sorted(models)

def get_mmproj_models():
    if not MMPROJ_DIR.exists():
        return []
    return sorted([f.name for f in MMPROJ_DIR.iterdir() if f.is_file() and f.suffix == ".gguf"])

def update_dropdowns():
    return gr.Dropdown(choices=get_inference_models()), gr.Dropdown(choices=get_mmproj_models())

def auto_select_mmproj(selected_model):
    if not selected_model:
        return gr.update(value=False), gr.update(value=None)
    
    model_filename = selected_model.split("/")[-1]
    mmproj_list = get_mmproj_models()
    for mmproj_file in mmproj_list:
        base_name = mmproj_file.replace("mmproj-", "").replace(".gguf", "")
        if base_name in model_filename:
            return gr.update(value=True), gr.update(value=mmproj_file)
    
    return gr.update(value=False), gr.update(value=None)

def start_server(
    model_name, backend_choice, ctx_len, 
    use_t, threads, use_tb, threads_batch, 
    use_ngl, ngl, use_ncmoe, ncmoe, 
    flash_attn, flash_attn_val, use_kv, ctk, ctv, 
    use_mmproj, mmproj_name, launch_chat, run_benchmark
):
    global server_process, chat_process
    
    if not model_name:
        yield "エラー: 推論モデルを選択してください。"
        return
        
    if use_mmproj and not mmproj_name:
        yield "エラー: マルチモーダルプロジェクタを使用する設定になっていますが、ファイルが選択されていません。"
        return

    if server_process is not None and server_process.poll() is None:
        server_process.terminate()
        server_process.wait()
        
    if chat_process is not None and chat_process.poll() is None:
        chat_process.terminate()
        chat_process.wait()

    # パス解決: MODELS_DIR を基点にする
    model_path = MODELS_DIR / model_name
    build_dir = "build-cpu" if backend_choice == "CPU" else "build-vulkan"

    # コマンドの組み立て
    if run_benchmark:
        bin_path = BASE_DIR / "llama.cpp" / build_dir / "bin" / "llama-bench"
        command = [str(bin_path), "-m", str(model_path)]
        
        if use_t:
            command.extend(["-t", str(threads)])
        if use_ngl:
            command.extend(["-ngl", str(ngl)])
            
        base_log = f"📊 ベンチマークを開始します...\n"
    else:
        bin_path = BASE_DIR / "llama.cpp" / build_dir / "bin" / "llama-server"
        # 基本コマンド (-c は常に指定)
        command = [
            str(bin_path),
            "-m", str(model_path),
            "-c", str(ctx_len)
        ]
        
        # チェックボックスがONのものだけパラメータを追加する
        if use_t:
            command.extend(["-t", str(threads)])
        if use_tb:
            command.extend(["-tb", str(threads_batch)])
        if use_ngl:
            command.extend(["-ngl", str(ngl)])
        if use_ncmoe:
            command.extend(["-ncmoe", str(ncmoe)])
        if flash_attn:
            command.extend(["-fa", str(flash_attn_val)])
        if use_kv:
            command.extend(["-ctk", str(ctk), "-ctv", str(ctv)])

        # 常に指定する固定パラメータ
        command.extend([
            "--port", "8080",
            "--host", "127.0.0.1",
            "--jinja",
            "--reasoning", "off"
        ])

        if not launch_chat:
            command.extend(["--alias", "local-model"])

        if use_mmproj:
            mmproj_path = MMPROJ_DIR / mmproj_name
            command.extend(["--mmproj", str(mmproj_path)])

        base_log = f"🚀 サーバーを起動しています...\n"

    yield base_log + f"実行コマンド: {' '.join(command)}\n\n起動準備中... ⏳"

    # ログファイルのパス (run.pyと同じ階層)
    log_file_path = BASE_DIR / "server_log.txt"

    try:
        server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        log_buffer = deque(maxlen=20)
        chat_opened = False 
        
        # ログファイルを書き込みモードで開く
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"実行コマンド: {' '.join(command)}\n\n")
            
            for line in server_process.stdout:
                # ファイルに書き込んで即座にフラッシュ
                log_file.write(line)
                log_file.flush()
                
                clean_line = line.strip()
                if clean_line:
                    log_buffer.append(clean_line)
                    yield base_log + "\n".join(log_buffer)
                    
                    if not run_benchmark and launch_chat and not chat_opened and "server is listening on" in clean_line:
                        # chat_script_path はツール本体側(BASE_DIR)を参照
                        chat_script_path = BASE_DIR / "python" / "chat.py"
                        chat_process = subprocess.Popen([sys.executable, str(chat_script_path)])
                        chat_opened = True
                
        server_process.wait()
        
        # 終了コードの記録
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            if run_benchmark:
                log_file.write(f"\n🏁 ベンチマーク完了 (Code: {server_process.returncode})\n")
            else:
                log_file.write(f"\n🛑 サーバー停止 (Code: {server_process.returncode})\n")

        if run_benchmark:
            yield base_log + "\n".join(log_buffer) + f"\n\n🏁 ベンチマーク完了 (Code: {server_process.returncode})"
        else:
            yield base_log + "\n".join(log_buffer) + f"\n\n🛑 サーバー停止 (Code: {server_process.returncode})"
        
    except Exception as e:
        # 例外発生時もログに記録
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n予期せぬエラーが発生しました:\n{str(e)}\n")
        yield f"予期せぬエラーが発生しました:\n{str(e)}"

def stop_server():
    global server_process, chat_process
    if server_process is not None and server_process.poll() is None:
        server_process.terminate()
        server_process.wait()
    if chat_process is not None and chat_process.poll() is None:
        chat_process.terminate()
        chat_process.wait()

# ==========================================
# Gradio UI
# ==========================================
with gr.Blocks(title="llama-server UI") as demo:
    gr.Markdown("## llama-server 起動ツール")
    
    with gr.Row():
        with gr.Column(scale=4):
            model_dropdown = gr.Dropdown(
                label="推論モデルを選択",
                choices=get_inference_models(),
                interactive=True
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 リスト更新")
            
    with gr.Row():
        backend_radio = gr.Radio(
            label="バックエンド",
            choices=["Vulkan", "CPU"],
            value="Vulkan",
            interactive=True
        )
        ctx_input = gr.Number(label="コンテキスト長 (-c)", value=32768, precision=0)

    # 詳細設定エリア
    with gr.Accordion("🛠️ 詳細パフォーマンス設定 (必要な場合のみチェック)", open=False):
        
        with gr.Row():
            use_t_cb = gr.Checkbox(label="-t を指定する", value=False)
            threads_input = gr.Number(label="生成スレッド数 (-t)", value=12, precision=0, interactive=False)

        with gr.Row():
            use_tb_cb = gr.Checkbox(label="-tb を指定する", value=False)
            threads_batch_input = gr.Number(label="バッチ処理スレッド数 (-tb)", value=24, precision=0, interactive=False)

        with gr.Row():
            use_ngl_cb = gr.Checkbox(label="-ngl を指定する", value=False)
            ngl_input = gr.Slider(0, 99, value=99, step=1, label="GPUオフロード層数 (-ngl)", interactive=False)
            
        with gr.Row():
            use_ncmoe_cb = gr.Checkbox(label="-ncmoe を指定する", value=False)
            ncmoe_input = gr.Slider(0, 99, value=99, step=1, label="MoEエキスパートオフロード (-ncmoe)", interactive=False)
        
        with gr.Row():
            use_kv_cb = gr.Checkbox(label="-ctk / -ctv を指定する", value=False)
            ctk_dropdown = gr.Dropdown(
                label="KVキャッシュ Key量子化 (-ctk)",
                choices=["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                value="q8_0",
                interactive=True
            )
            ctv_dropdown = gr.Dropdown(
                label="KVキャッシュ Value量子化 (-ctv)",
                choices=["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                value="q8_0",
                interactive=True
            )

        with gr.Row():
            flash_attn_cb = gr.Checkbox(label="Flash Attentionを指定 (-fa)", value=False)
            flash_attn_dropdown = gr.Dropdown(
                label="設定値",
                choices=["on", "off", "auto"],
                value="on",
                interactive=False
            )

    with gr.Row():
        use_mmproj_cb = gr.Checkbox(label="マルチモーダルプロジェクタを使用 (--mmproj)", value=False)
        mmproj_dropdown = gr.Dropdown(label="mmprojファイルを選択", choices=get_mmproj_models())

    with gr.Row():
        launch_chat_cb = gr.Checkbox(label="チャットUI (chat.py) も起動", value=False)
        benchmark_cb = gr.Checkbox(label="📊 ベンチマークモード", value=False)

    with gr.Row():
        start_btn = gr.Button("▶️ 起動 / 実行", variant="primary")
        stop_btn = gr.Button("⏹️ 停止", variant="stop")
        
    output_log = gr.Textbox(label="実行状況", lines=22)

    # --- UIのインタラクティブ制御 ---
    use_t_cb.change(fn=lambda x: gr.update(interactive=x), inputs=use_t_cb, outputs=threads_input)
    use_tb_cb.change(fn=lambda x: gr.update(interactive=x), inputs=use_tb_cb, outputs=threads_batch_input)
    use_ngl_cb.change(fn=lambda x: gr.update(interactive=x), inputs=use_ngl_cb, outputs=ngl_input)
    use_ncmoe_cb.change(fn=lambda x: gr.update(interactive=x), inputs=use_ncmoe_cb, outputs=ncmoe_input)
    use_kv_cb.change(
        fn=lambda x: [gr.update(interactive=x), gr.update(interactive=x)], 
        inputs=use_kv_cb, 
        outputs=[ctk_dropdown, ctv_dropdown]
    )
    flash_attn_cb.change(fn=lambda x: gr.update(interactive=x), inputs=flash_attn_cb, outputs=flash_attn_dropdown)

    refresh_btn.click(fn=update_dropdowns, outputs=[model_dropdown, mmproj_dropdown])
    model_dropdown.change(fn=auto_select_mmproj, inputs=model_dropdown, outputs=[use_mmproj_cb, mmproj_dropdown])

    start_btn.click(
        fn=start_server,
        inputs=[
            model_dropdown, backend_radio, ctx_input, 
            use_t_cb, threads_input, use_tb_cb, threads_batch_input, 
            use_ngl_cb, ngl_input, use_ncmoe_cb, ncmoe_input, 
            flash_attn_cb, flash_attn_dropdown, use_kv_cb, ctk_dropdown, ctv_dropdown, 
            use_mmproj_cb, mmproj_dropdown, launch_chat_cb, benchmark_cb
        ],
        outputs=output_log
    )
    
    stop_btn.click(fn=stop_server)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True)
