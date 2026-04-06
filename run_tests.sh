#!/bin/bash

set -e

# ================= 設定區域 =================
DATASETS="car tic-tac-toe"
MINSUPS_LIST="100 90 80 70 60 50 40 30 20 10 5 2 1 0.5"
BASELINES="PHA"

PHA_BIN="tools/pha_fim"
PHA_MAIN="src/main.cpp"
PHA_CU="cuda/pha_fim_gpu.cu"
INCLUDE_FLAGS="-Iinclude"

NVCC_BIN="${NVCC_BIN:-nvcc}"
# ===========================================

MINSUPS_CMD_ARG=$(echo "$MINSUPS_LIST" | tr ' ' ',')

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

echo "========================================"
echo "開始自動化測試流程"
echo "Datasets : $DATASETS"
echo "MinSups  : $MINSUPS_CMD_ARG"
echo "Baselines: $BASELINES"
echo "ARCH     : $ARCH_FLAG"
echo "========================================"

echo ""
echo "[Step 0] Building PHA..."

mkdir -p tools

$NVCC_BIN -O2 -std=c++17 \
    $ARCH_FLAG \
    "$PHA_MAIN" "$PHA_CU" \
    $INCLUDE_FLAGS \
    -o "$PHA_BIN"

if [ ! -f "$PHA_BIN" ]; then
    echo "❌ Error: PHA binary not found after build: $PHA_BIN"
    exit 1
fi

chmod +x "$PHA_BIN"
echo "✅ Build success: $PHA_BIN"

export PHA_BIN="$(pwd)/$PHA_BIN"

for dataset in $DATASETS; do
    echo ""
    echo ">>> 正在處理 Dataset: $dataset"
    echo "[Step 1] Running experiment.py..."

    python3 experiment.py \
        --baselines "$BASELINES" \
        --datasets "$dataset" \
        --minsup-ratios "$MINSUPS_CMD_ARG" \
        --tx-ratios "100" \
        --override-default-minsup "$dataset=100" \
        --keep-pattern-files \
        --resume
done

echo ""
echo "========================================"
echo "所有測試結束"
echo "========================================"
