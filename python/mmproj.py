import gradio as gr
import subprocess
import os
from pathlib import Path

# ==========================================
# ディレクトリ設定 (config.yamlから読み込み)
# ==========================================
from python.config import BASE_DIR, MODELS_DIR

HF_MODELS_DIR = MODELS_DIR / "hf_models"
MMPROJ_DIR = MODELS_DIR / "mmproj"
# ==========================================

def get_model_list():
    """
    ダウンロード済みモデルディレクトリの一覧を取得する関数
    """
    if not HF_MODELS_DIR.exists():
        return []
    return [d.name for d in HF_MODELS_DIR.iterdir() if d.is_dir()]

def update_dropdown():
    return gr.Dropdown(choices=get_model_list())

def convert_mmproj(model_name):
    """
    選択されたモデルのマルチモーダルプロジェクタを抽出・変換する関数
    """
    if not model_name:
        yield "エラー: プルダウンからモデルを選択してください。"
        return

    try:
        model_path = HF_MODELS_DIR / model_name
        
        # 出力先のフォルダが存在しなければ作成
        MMPROJ_DIR.mkdir(parents=True, exist_ok=True)
        
        # 出力ファイル名は「mmproj-モデル名.gguf」とする
        outfile_path = MMPROJ_DIR / f"mmproj-{model_name}.gguf"

        # ========== 既存ファイルのスキップ判定 ==========
        if outfile_path.exists():
            yield f"⚠️ スキップしました\nすでに以下のプロジェクタファイルが存在します。\nパス: {outfile_path}"
            return
        # ================================================

        convert_script = BASE_DIR / "llama.cpp" / "convert_hf_to_gguf.py"

        # mmproj変換の実行コマンド組み立て
        command = [
            "python",
            str(convert_script),
            str(model_path),
            "--mmproj", # プロジェクタのみを抽出・変換するフラグ
            "--outfile",
            str(outfile_path)
        ]

        base_log = f"マルチモーダルプロジェクタ(mmproj)の変換を開始します...\n対象: {model_name}\n出力先: {outfile_path}\n\n[実行進捗]\n"
        yield base_log + "モデルを解析中... プロジェクタ部分を抽出しています ⏳"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            # INFO や WARNING ログはスキップ
            if "INFO:" in line or "WARNING:" in line:
                continue
            
            # プログレスバーの行だけを拾う
            if "Writing:" in line:
                clean_line = line.replace('\r', '').replace('\n', '').strip()
                if clean_line:
                    yield base_log + clean_line

        process.wait()

        if process.returncode == 0:
            yield base_log + f"Writing: 100%|██████████| 完了！\n\n✅ mmproj変換完了！\n保存先: {outfile_path}"
        else:
            # エラー時、プロジェクタが含まれていないモデルの可能性も示唆する
            error_msg = f"\n❌ エラーが発生しました (終了コード: {process.returncode})\n"
            error_msg += "※このモデルにはマルチモーダルプロジェクタが含まれていない、または対応していない形式の可能性があります。"
            yield base_log + error_msg

    except Exception as e:
        yield f"予期せぬエラーが発生しました:\n{str(e)}"

# ==========================================
# Gradio UI の構築
# ==========================================
with gr.Blocks(title="mmproj Converter UI") as demo:
    gr.Markdown("## マルチモーダルプロジェクタ(mmproj)抽出ツール")
    gr.Markdown("VLMモデルから、画像解析用プロジェクタを `mmproj-*.gguf` として抽出・変換します。")
    
    with gr.Row():
        with gr.Column(scale=4):
            model_dropdown = gr.Dropdown(
                label="変換するモデルを選択",
                choices=get_model_list(),
                interactive=True
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 リスト更新")
    
    convert_btn = gr.Button("mmprojを抽出・変換", variant="primary")
    output_log = gr.Textbox(label="実行ログ", lines=10)

    refresh_btn.click(
        fn=update_dropdown,
        inputs=None,
        outputs=model_dropdown
    )

    convert_btn.click(
        fn=convert_mmproj,
        inputs=model_dropdown,
        outputs=output_log
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True)
