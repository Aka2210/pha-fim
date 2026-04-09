#!/bin/bash

set -euo pipefail

# ================= 設定區域 =================
DATASETS="car tic-tac-toe"

# tx-ratio sweep 的固定 minsup 百分比
# 在 --tx-sweep-minsup-mode count 下，
# experiment.py 會先用這個百分比 * full dataset size 算出固定 minsup_count
TX_SWEEP_BASE_MINSUP_PERCENT="${TX_SWEEP_BASE_MINSUP_PERCENT:-5}"

# tx sweep 的 x 軸
TX_RATIOS="${TX_RATIOS:-10,20,30,50,70,100}"

# minsup sweep 的 x 軸（若你只想看 tx sweep，可先留著）
MINSUPS_LIST="${MINSUPS_LIST:-100 90 80 70 60 50 40 30 20 10 5 2 1 0.5}"

BASELINES="${BASELINES:-FPGrowth_itemsets,PHA}"

PHA_BIN="tools/pha_fim"
PHA_MAIN="src/main.cpp"
PHA_CU="cuda/pha_fim_gpu.cu"
INCLUDE_FLAGS="-Iinclude"

NVCC_BIN="${NVCC_BIN:-nvcc}"

# ---- 資源限制（保守值）----
JOBS="${JOBS:-4}"
OMP_THREADS="${OMP_THREADS:-8}"

# 啟用學長說的 count 模式
TX_SWEEP_MINSUP_MODE="${TX_SWEEP_MINSUP_MODE:-count}"

# 可手動指定 GPU；不指定則自動挑最空的一張
CUDA_DEVICE="${CUDA_DEVICE:-}"

MAX_CPU_USAGE_PERCENT=50
MAX_MEM_USAGE_PERCENT=50
MAX_GPU_MEM_USAGE_PERCENT=50
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

pick_idle_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return 0
    fi

    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | sort -t',' -k2,2n \
        | head -n 1 \
        | awk -F',' '{gsub(/ /, "", $1); print $1}'
}

check_system_load() {
    echo "[Check] CPU / Memory / GPU usage before run"

    if command -v free >/dev/null 2>&1; then
        MEM_TOTAL=$(free -m | awk '/Mem:/ {print $2}')
        MEM_USED=$(free -m | awk '/Mem:/ {print $3}')
        if [[ -n "${MEM_TOTAL:-}" && "$MEM_TOTAL" -gt 0 ]]; then
            MEM_USED_PCT=$(( MEM_USED * 100 / MEM_TOTAL ))
        else
            MEM_USED_PCT=0
        fi
    else
        MEM_USED_PCT=0
    fi

    LOAD_AVG=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo "N/A")
    CPU_THREADS=$(nproc 2>/dev/null || echo "N/A")

    echo "  - CPU threads available : $CPU_THREADS"
    echo "  - Load average (1 min)  : $LOAD_AVG"
    echo "  - Memory used           : ${MEM_USED_PCT}%"

    if [[ "$MEM_USED_PCT" -ge "$MAX_MEM_USAGE_PERCENT" ]]; then
        echo "⚠️  Memory usage is already >= ${MAX_MEM_USAGE_PERCENT}%"
        echo "    建議先用 htop / free -h 確認是否有人在用。"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  - GPU status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

        while IFS=',' read -r idx mem_used mem_total; do
            idx=$(echo "$idx" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            if [[ "$mem_total" -gt 0 ]]; then
                pct=$(( mem_used * 100 / mem_total ))
                if [[ "$pct" -ge "$MAX_GPU_MEM_USAGE_PERCENT" ]]; then
                    echo "⚠️  GPU $idx memory usage is already >= ${MAX_GPU_MEM_USAGE_PERCENT}%"
                fi
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    fi
}

ARCH_FLAG=$(detect_cuda_arch_flag)

export OMP_NUM_THREADS="$OMP_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_THREADS"

if [[ -z "$CUDA_DEVICE" ]]; then
    CUDA_DEVICE=$(pick_idle_gpu)
fi

if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

echo "========================================"
echo "開始自動化測試流程"
echo "Datasets : $DATASETS"
echo "Baselines: $BASELINES"
echo "ARCH     : $ARCH_FLAG"
echo "JOBS     : $JOBS"
echo "OMP_THD  : $OMP_THREADS"
echo "GPU      : ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "TX mode  : $TX_SWEEP_MINSUP_MODE"
echo "TX minsup: ${TX_SWEEP_BASE_MINSUP_PERCENT}% of full dataset"
echo "TX ratios: $TX_RATIOS"
echo "MS sweep : $MINSUPS_CMD_ARG"
echo "========================================"

check_system_load

echo ""
echo "[Step 0] Building PHA..."

mkdir -p tools

rm -f "$PHA_BIN"
"$NVCC_BIN" -O2 -std=c++17 \
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
        --tx-ratios "$TX_RATIOS" \
        --tx-sweep-minsup-mode "$TX_SWEEP_MINSUP_MODE" \
        --override-default-minsup "$dataset=$TX_SWEEP_BASE_MINSUP_PERCENT" \
        --keep-pattern-files \
        --jobs "$JOBS"
done

echo ""
echo "========================================"
echo "所有測試結束"
echo "========================================"
echo "建議執行中另開視窗觀察："
echo "  htop"
echo "  watch -n 2 nvidia-smi"
