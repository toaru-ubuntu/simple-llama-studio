import gradio as gr
import subprocess
import os
from pathlib import Path

# ==========================================
# ディレクトリ設定 (config.yamlから読み込み)
# ==========================================
from python.config import BASE_DIR, MODELS_DIR

# ==========================================

download_process = None

def download_model(hf_id, include_pattern, use_safe_mode):
    """
    ダウンロードを実行する関数
    """
    global download_process

    if not hf_id or "/" not in hf_id:
        yield "エラー: 有効なモデルIDを指定または入力してください。"
        return
    
    if download_process is not None and download_process.poll() is None:
        download_process.terminate()
        download_process.wait()

    try:
        model_name = hf_id.split("/")[-1]
        save_dir = MODELS_DIR / "hf_models" / model_name
        
        (MODELS_DIR / "hf_models").mkdir(parents=True, exist_ok=True)
        
        # 基本コマンド
        command = [
            "hf", 
            "download", 
            hf_id, 
            "--local-dir", 
            str(save_dir)
        ]
        
        # 特定のファイル/フォルダのみを含める（--include）
        if include_pattern.strip():
            patterns = include_pattern.strip().split()
            for pattern in patterns:
                command.extend(["--include", pattern])
        
        # 実行環境の変数をコピー
        env = os.environ.copy()
        
        # セーフモード（単一ファイルずつの安定ダウンロード）の適用
        mode_text = "標準モード (高速/並列)"
        if use_safe_mode:
            mode_text = "セーフモード (安定/単一ファイル)"
            env["HF_HUB_DISABLE_XET"] = "1"
            command.extend(["--max-workers", "1"])
        
        base_log = f"ダウンロードを開始します...\n対象: {hf_id}\n絞り込み: {include_pattern or '指定なし (全て)'}\nモード: {mode_text}\n保存先: {save_dir}\n\n[実行進捗]\n"
        yield base_log + "通信を確立中... ダウンロードの準備をしています ⏳"
        
        # subprocess.Popen から bufsize=1 などを削除し、プレーンに立ち上げます
        download_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            env=env
        )
        
        # === 修正箇所: 1文字ずつ読み込み、\r でも画面を更新する ===
        char_buffer = []
        while True:
            char = download_process.stdout.read(1)
            
            # プロセスが終了し、読み込む文字がなくなったらループを抜ける
            if not char and download_process.poll() is not None:
                break
                
            # \r (カーソル戻し) か \n (改行) が来たら、そこまでの文字列を出力する
            if char in ('\r', '\n'):
                clean_line = "".join(char_buffer).strip()
                char_buffer.clear()
                if clean_line:
                    yield base_log + clean_line
            elif char:
                char_buffer.append(char)
        # =========================================================

        download_process.wait()
        
        if download_process.returncode == 0:
            yield base_log + f"100% 完了！\n\n✅ ダウンロード完了！\n保存先: {save_dir}"
        else:
            yield base_log + f"\n🛑 ダウンロードが停止されたか、エラーが発生しました (終了コード: {download_process.returncode})"
            
    except Exception as e:
        yield f"予期せぬシステムエラーが発生しました:\n{str(e)}"

def stop_download():
    """停止ボタンの処理"""
    global download_process
    if download_process is not None and download_process.poll() is None:
        download_process.terminate()
        download_process.wait()

# ==========================================
# Gradio UI の構築
# ==========================================
with gr.Blocks(title="HF Model Downloader") as demo:
    gr.Markdown("## Hugging Face モデルダウンローダー")
    
    with gr.Row():
        with gr.Column(scale=2):
            hf_id_input = gr.Textbox(
                label="Hugging Face Model ID", 
                placeholder="例: Qwen/Qwen3-Coder-Next-GGUF",
                value="" 
            )
            
            include_input = gr.Textbox(
                label="絞り込みパターン (特定のファイルのみ落としたい場合)", 
                placeholder="例: Qwen3-Coder-Next-Q4_K_M/* または  *.safetensors",
                value="",
                info="空白の場合はリポジトリ全体をダウンロードします。複数指定する場合はスペースで区切ってください。"
            )
            
        with gr.Column(scale=1):
            safe_mode_checkbox = gr.Checkbox(
                label="セーフモードでダウンロード",
                value=True,
                info="通信が途切れる場合はチェック。Xetを無効化し、1ファイルずつ確実に保存します。"
            )

    with gr.Row():
        download_btn = gr.Button("▶️ ダウンロード実行", variant="primary")
        stop_btn = gr.Button("⏹️ 停止", variant="stop")
    
    output_log = gr.Textbox(label="実行ログ", lines=10)
    
    download_btn.click(
        fn=download_model,
        inputs=[hf_id_input, include_input, safe_mode_checkbox],
        outputs=output_log
    )

    stop_btn.click(
        fn=stop_download,
        inputs=None,
        outputs=None
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True)
