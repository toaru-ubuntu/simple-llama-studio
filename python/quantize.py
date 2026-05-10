import gradio as gr
import subprocess
import os
from pathlib import Path

# ==========================================
# ディレクトリ設定 (config.yamlから読み込み)
# ==========================================
from python.config import BASE_DIR, MODELS_DIR

GGUF_DIR = MODELS_DIR / "gguf"
QUANTIZE_DIR = MODELS_DIR / "quantize"
# ==========================================

# 量子化形式のリスト（プルダウン用）
QUANT_TYPES = [
    "Q2_K",
    "Q3_K_L",
    "Q4_0",
    "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q8_0"
]

def get_gguf_list():
    if not GGUF_DIR.exists():
        return []
    
    return [f.name for f in GGUF_DIR.iterdir() if f.is_file() and f.suffix == ".gguf"]

def update_dropdown():
    return gr.Dropdown(choices=get_gguf_list())

def quantize_model(gguf_filename, quant_type):
    if not gguf_filename:
        yield "エラー: 変換元のGGUFモデルを選択してください。"
        return
    if not quant_type:
        yield "エラー: 量子化の形式(Q4_K_Mなど)を選択してください。"
        return

    try:
        input_path = GGUF_DIR / gguf_filename
        QUANTIZE_DIR.mkdir(parents=True, exist_ok=True)
        
        base_name = input_path.stem
        output_filename = f"{base_name}-{quant_type}.gguf"
        output_path = QUANTIZE_DIR / output_filename

        # ========== 既存ファイルのスキップ判定 ==========
        if output_path.exists():
            yield f"⚠️ スキップしました\nすでに以下のファイルが存在するため、量子化処理をスキップします。\nパス: {output_path}"
            return
        # ================================================

        # llama-quantizeの実行パスも BASE_DIR を基準に指定
        quantize_bin = BASE_DIR / "llama.cpp" / "build-cpu" / "bin" / "llama-quantize"

        command = [
            str(quantize_bin),
            str(input_path),
            str(output_path),
            quant_type
        ]

        base_log = f"量子化を開始します...\n対象: {gguf_filename}\n形式: {quant_type}\n出力先: {output_path}\n\n[実行進捗]\n"
        yield base_log + "モデルを読み込み中... ⏳"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            clean_line = line.replace('\r', '').replace('\n', '').strip()
            if not clean_line:
                continue
            
            if clean_line.startswith("[") and "]" in clean_line:
                yield base_log + clean_line

        process.wait()

        if process.returncode == 0:
            yield base_log + f"100% 完了！\n\n✅ 量子化完了！\n保存先: {output_path}"
        else:
            yield base_log + f"\n❌ エラーが発生しました (終了コード: {process.returncode})"

    except Exception as e:
        yield f"予期せぬエラーが発生しました:\n{str(e)}"

# ==========================================
# Gradio UI の構築
# ==========================================
with gr.Blocks(title="GGUF Quantizer UI") as demo:
    gr.Markdown("## llama.cpp 量子化ツール")
    gr.Markdown("指定した形式で軽量化して保存します。")
    
    with gr.Row():
        with gr.Column(scale=4):
            gguf_dropdown = gr.Dropdown(
                label="元となるGGUFモデルを選択",
                choices=get_gguf_list(),
                interactive=True
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 リスト更新")
            
    with gr.Row():
        quant_dropdown = gr.Dropdown(
            label="量子化形式を選択",
            choices=QUANT_TYPES,
            value="Q4_K_M",
            interactive=True
        )
    
    quantize_btn = gr.Button("量子化を実行", variant="primary")
    output_log = gr.Textbox(label="実行ログ", lines=9)

    refresh_btn.click(
        fn=update_dropdown,
        inputs=None,
        outputs=gguf_dropdown
    )

    quantize_btn.click(
        fn=quantize_model,
        inputs=[gguf_dropdown, quant_dropdown],
        outputs=output_log
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True)
