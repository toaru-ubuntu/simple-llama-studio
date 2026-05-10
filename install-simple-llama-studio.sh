#!/bin/bash


T_DIR=$HOME/install/simple-llama-studio

mkdir -p ${T_DIR}
cp run.py ${T_DIR}
cp -r python ${T_DIR}

cd ${T_DIR}

uv venv --python 3.12
source .venv/bin/activate

uv pip install huggingface_hub gradio pyyaml

#準備とllama.cppのソースのダウンロード
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
