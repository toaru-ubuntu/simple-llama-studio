# python/config.py
import os
import yaml
from pathlib import Path

# =========================================================
# 【重要】動的パスの取得
# __file__ はこのファイル(config.py)の絶対パスを表します。
# .resolve().parent.parent とすることで、
# pythonディレクトリのさらに1つ上（simple-llama-studio）を
# どこにインストールされていても自動的に取得します。
# =========================================================
STUDIO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = STUDIO_ROOT / "config.yaml"

def load_config():
    # デフォルト設定（自動取得したルートディレクトリを基準にする）
    config_data = {
        "base_path": str(STUDIO_ROOT),
        "models_path": str(STUDIO_ROOT / "models")
    }
    
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                user_config = yaml.safe_load(f)
                if user_config and "simple_llama_studio" in user_config:
                    app_config = user_config["simple_llama_studio"]
                    
                    # 万が一ユーザーが base_path を強制指定した場合は上書き
                    if "base_path" in app_config and app_config["base_path"]:
                        config_data["base_path"] = os.path.expandvars(app_config["base_path"])
                        
                    # models_path の指定がある場合
                    if "models_path" in app_config and app_config["models_path"]:
                        m_path = os.path.expandvars(app_config["models_path"])
                        # 絶対パス (例: /mnt/data/models) か、相対パス (例: models) かを判定
                        if Path(m_path).is_absolute():
                            config_data["models_path"] = m_path
                        else:
                            # 相対パスの場合は base_path と結合する
                            config_data["models_path"] = str(Path(config_data["base_path"]) / m_path)
                            
            except Exception as e:
                print(f"⚠️ config.yaml の読み込みに失敗しました。デフォルト設定を使用します: {e}")
                
    return config_data

# 読み込んだ設定を変数としてエクスポート
settings = load_config()
BASE_DIR = Path(settings["base_path"])
MODELS_DIR = Path(settings["models_path"])
