#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
NC='\033[0m'

mkdir -p "$SCRIPT_DIR/data_raw" "$SCRIPT_DIR/tools" "$SCRIPT_DIR/results"

echo -e "${GREEN}[0/4] Checking project structure...${NC}"
if [ ! -f "$SCRIPT_DIR/src/main.cpp" ]; then
    echo "❌ src/main.cpp not found."
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/cuda/pha_fim_gpu.cu" ]; then
    echo "❌ cuda/pha_fim_gpu.cu not found."
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/include/pha_fim_gpu.hpp" ]; then
    echo "❌ include/pha_fim_gpu.hpp not found."
    exit 1
fi

if ! command -v wget &> /dev/null; then
    echo "⚠️ wget could not be found. Please install wget manually if needed."
fi

if [ ! -f "$SCRIPT_DIR/tools/spmf.jar" ]; then
    echo "Downloading spmf.jar..."
    wget https://www.philippe-fournier-viger.com/spmf/spmf.jar -O "$SCRIPT_DIR/tools/spmf.jar"
else
    echo "spmf.jar already exists in tools/."
fi

echo -e "${GREEN}[1/4] Checking System Dependencies...${NC}"

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 could not be found. Please install it."
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc could not be found. Please install CUDA toolkit or load the CUDA module."
    exit 1
fi

echo -e "${GREEN}[2/4] Setting up Python Virtual Environment...${NC}"
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    python3 -m venv "$SCRIPT_DIR/.venv"
    echo "Created virtual environment in .venv"
else
    echo "Virtual environment already exists."
fi

source "$SCRIPT_DIR/.venv/bin/activate"

echo -e "${GREEN}[3/4] Installing Python Dependencies...${NC}"
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo -e "${GREEN}[4/4] Compiling CUDA Project...${NC}"

# 可由外部覆寫，例如：
# NVCC_BIN=/usr/local/cuda-12.8/bin/nvcc ./setup_env.sh
NVCC_BIN="${NVCC_BIN:-nvcc}"

# 自動抓 GPU compute capability；抓不到就預設 sm_70
detect_cuda_arch_flag() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')
        if [[ "$CC" =~ ^[0-9]+\.[0-9]+$ ]]; then
            MAJOR="${CC%%.*}"
            MINOR="${CC##*.}"
            SM="${MAJOR}${MINOR}"
            echo "-arch=sm_${SM}"
            return 0
        fi
    fi
    echo "-arch=sm_70"
}

ARCH_FLAG=$(detect_cuda_arch_flag)

echo "Using NVCC     : $NVCC_BIN"
echo "Using ARCH FLAG: $ARCH_FLAG"

"$NVCC_BIN" -O2 -std=c++17 \
    $ARCH_FLAG \
    "$SCRIPT_DIR/src/main.cpp" \
    "$SCRIPT_DIR/cuda/pha_fim_gpu.cu" \
    -I"$SCRIPT_DIR/include" \
    -o "$SCRIPT_DIR/tools/pha_fim"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Environment Setup Complete!${NC}"
    echo "Binary built at: $SCRIPT_DIR/tools/pha_fim"
    echo "To activate environment manually, run:"
    echo "  source .venv/bin/activate"
else
    echo "❌ Compilation failed."
    exit 1
fi
