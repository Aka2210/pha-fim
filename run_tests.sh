#!/bin/bash

# ================= 設定區域 =================
# 1. 設定你要測試的 Datasets (以空白分隔)
# DATASETS="car mushroom kr-vs-kp tic-tac-toe"
DATASETS="car tic-tac-toe"

# 2. 設定你要測試的 MinSup Ratios (以空白分隔)
# 注意：experiment.py 會將這些數字轉為 float (例如 5 -> 5.0)，腳本會自動處理檔名對應
MINSUPS_LIST="100 90 80 70 60 50 40 30 20 10 5 2 1 0.5"

# 3. 設定 Baselines (固定為這兩個以便驗證)
BASELINES="FPGrowth_itemsets,Hamm"

# ===========================================

# 將空白分隔的 minsups 轉換為逗號分隔 (給 experiment.py 使用)
# 也就是將 "0.1 0.5 1" 轉為 "0.1,0.5,1"
MINSUPS_CMD_ARG=$(echo $MINSUPS_LIST | tr ' ' ',')

echo "========================================"
echo "開始自動化測試流程"
echo "Datasets: $DATASETS"
echo "MinSups : $MINSUPS_CMD_ARG"
echo "========================================"

for dataset in $DATASETS; do
    echo ""
    echo ">>> 正在處理 Dataset: $dataset"

    # ---------------------------------------------------------
    # 步驟 1: 執行 experiment.py 產生結果檔案
    # ---------------------------------------------------------
    echo "[Step 1] Running experiment.py..."
    # for ms in $MINSUPS_LIST; do
        echo "   - Preparing for MinSup: $ms"
        python3 experiment.py \
            --baselines "$BASELINES" \
            --datasets "$dataset" \
            --minsup-ratios "$MINSUPS_CMD_ARG" \
            --tx-ratios "100" \
            --override-default-minsup "$dataset=100" \
            --keep-pattern-files \
            --resume
    # done

    if [ $? -ne 0 ]; then
        echo "❌ Error: experiment.py 執行失敗，跳過 $dataset 的驗證。"
        continue
    fi


done

# ---------------------------------------------------------
# 步驟 2: 針對每個 MinSup 執行 verify.py
# ---------------------------------------------------------
echo "[Step 2] Verifying results..."
for dataset in $DATASETS; do
    echo ""
    echo ">>> 驗證 Dataset: $dataset"
    for ms in $MINSUPS_LIST; do
        # 技巧：利用 Python 確保檔名格式與 experiment.py 一致 (例如 5 轉成 5.0)
        # 這是因為 experiment.py 內部使用了 float() 轉換
        ms_formatted=$(python3 -c "print(f'{float($ms)}')")

        # 定義檔案路徑
        FILE_HAMM="results/${dataset}/Hamm_ms${ms_formatted}.spmf"
        FILE_FP="results/${dataset}/FPGrowth_itemsets_ms${ms_formatted}.spmf"

        # 檢查檔案是否存在
        if [[ -f "$FILE_HAMM" && -f "$FILE_FP" ]]; then
            echo -n "   - Checking MinSup $ms_formatted ... "

            # 執行 verify.py (假設 verify.py 輸出結果到 stdout)
            # 這裡你可以決定是否要將輸出導向到檔案，目前直接印在螢幕上
            python3 verify.py "$FILE_HAMM" "$FILE_FP"

            # 檢查 verify.py 的回傳值 (如果 verify.py 寫得好，錯誤時應該回傳非0)
            if [ $? -eq 0 ]; then
                echo "✅ Done."
            else
                echo "⚠️  Verification Failed!"
            fi
        else
            echo "❌ Missing files for MinSup $ms_formatted (Expected: $FILE_HAMM)"
        fi
    done
done
echo ""
echo "========================================"
echo "所有測試結束"
echo "========================================"
