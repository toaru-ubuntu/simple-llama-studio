import gradio as gr
import subprocess
import os
from pathlib import Path

# ==========================================
# ディレクトリ設定 (config.yamlから読み込み)
# ==========================================
from python.config import BASE_DIR, MODELS_DIR

HF_MODELS_DIR = MODELS_DIR / "hf_models"
GGUF_DIR = MODELS_DIR / "gguf"
# ==========================================

def get_model_list():
    if not HF_MODELS_DIR.exists():
        return []
    return [d.name for d in HF_MODELS_DIR.iterdir() if d.is_dir()]

def update_dropdown():
    return gr.Dropdown(choices=get_model_list())

def convert_to_gguf(model_name):
    if not model_name:
        yield "エラー: プルダウンからモデルを選択してください。"
        return

    try:
        model_path = HF_MODELS_DIR / model_name
        GGUF_DIR.mkdir(parents=True, exist_ok=True)
        outfile_path = GGUF_DIR / f"{model_name}.gguf"

        # ========== 既存ファイルのスキップ判定 ==========
        if outfile_path.exists():
            yield f"⚠️ スキップしました\nすでに以下のファイルが存在するため、変換処理をスキップします。\nパス: {outfile_path}"
            return
        # ====================================================

        # convert_hf_to_gguf.py のパスも BASE_DIR を基準に指定
        convert_script = BASE_DIR / "llama.cpp" / "convert_hf_to_gguf.py"

        command = [
            "python",
            str(convert_script),
            str(model_path),
            "--outfile",
            str(outfile_path)
        ]

        # 常に表示させておく固定のテキスト
        base_log = f"GGUF変換を開始します...\n対象: {model_name}\n出力先: {outfile_path}\n\n[実行進捗]\n"
        
        # 変換初期のINFOが流れている間はプログレスバーが出ないので、待機メッセージを出しておく
        yield base_log + "モデルを解析・準備中... (1〜2分かかる場合があります) ⏳"

        # ログファイルのパスを定義
        log_file_path = BASE_DIR / "convert_log.txt"

        # subprocessを実行し、ログファイルに書き込みながら出力を処理
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                # ログファイルにすべての出力を書き込み、リアルタイムで保存
                log_file.write(line)
                log_file.flush()

                # 1. 大量の INFO や WARNING ログはすべて無視する（UI表示からはスキップ）
                if "INFO:" in line or "WARNING:" in line:
                    continue
                
                # 2. "Writing:" が含まれるプログレスバーの行だけを拾う
                if "Writing:" in line:
                    # ターミナル特有の改行コード(\rや\n)を消して、純粋な1行の文字列にする
                    clean_line = line.replace('\r', '').replace('\n', '').strip()
                    
                    # 蓄積（+=）ではなく、固定テキストに最新の1行だけを結合してGradioに渡す
                    if clean_line:
                        yield base_log + clean_line

            process.wait()

        # 最終結果の表示
        if process.returncode == 0:
            yield base_log + f"Writing: 100%|██████████| 完了！\n\n✅ 変換完了！\n保存先: {outfile_path}"
        else:
            yield base_log + f"\n❌ エラーが発生しました (終了コード: {process.returncode})\n詳細は {log_file_path} を確認してください。"

    except Exception as e:
        yield f"予期せぬエラーが発生しました:\n{str(e)}"

# ==========================================
# Gradio UI の構築
# ==========================================
with gr.Blocks(title="GGUF Converter UI") as demo:
    gr.Markdown("## llama.cpp GGUF 変換ツール")
    gr.Markdown("ダウンロードしたモデルを選択して、GGUF形式に変換します。")
    
    with gr.Row():
        with gr.Column(scale=4):
            model_dropdown = gr.Dropdown(
                label="変換するモデルを選択",
                choices=get_model_list(),
                interactive=True
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 リスト更新")
    
    convert_btn = gr.Button("GGUFに変換", variant="primary")
    
    output_log = gr.Textbox(label="実行ログ", lines=10)

    refresh_btn.click(
        fn=update_dropdown,
        inputs=None,
        outputs=model_dropdown
    )

    convert_btn.click(
        fn=convert_to_gguf,
        inputs=model_dropdown,
        outputs=output_log
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True)
