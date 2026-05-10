# simple-llama-studio
llama.cppをgradioGUIで操作できるようにしました。

## ✨ 主な機能
huggingfaceからモデルファイルのダウンロード。<br>
safetensorsからggufへデータ変換。<br>
ggufの量子化。<br>
mmprojの作成。<br>
llama-serverでの立ち上げ。<br>

## 💻 動作環境
* **OS**: Linux (Ubuntu26.04)で確認しています。
* **Python**: 3.12
* **ハードウェア**: 
  * 推奨: CPU,vulkanに対応したハードウェア

## 🛠 インストール方法
1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/toaru-ubuntu/simple-llama-studio.git
   cd simple-llama-studio
   
2. **uv仮想環境の作成と有効化**
    ```bash
    sudo apt install curl

    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    uv venv .venv --python 3.12
    source .venv/bin/activate

3. **必要なパッケージのインストール**
    ```bash
    uv pip install huggingface_hub gradio pyyaml
    
4. **準備とllama.cppのソースのダウンロード** 
    ```bash
    sudo apt install -y git cmake g++ libcurlpp-dev
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
    
    #CPU用ビルド（ggufの変換で使う）
    mkdir build-cpu
    cmake -B build-cpu
    cmake --build build-cpu --config Release -j$(nproc)

    #vulkanビルド
    sudo apt install -y libvulkan-dev glslc spirv-headers
    cmake -B build-vulkan -DGGML_VULKAN=ON
    cmake --build build-vulkan --config Release -j$(nproc)

    uv pip install -r ./requirements.txt --index-strategy unsafe-best-match
    
# 注意事項
* **使用するLLMについて**<br>
各LLMの注意事項に従ってください。

* **ライセンス**<br>
simple-llama-studio自体はMITライセンスですが、<br>
[llama.cppについてはこちら](https://github.com/ggml-org/llama.cpp)を確認してください。
