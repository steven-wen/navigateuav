#!/bin/bash
# Function：Bearing-UAV (cvphr) model training.
# Operation：
#   cd /your/path/of/proj/bearinguav
#   chmod +x ./scripts/cvphr_train.sh
#   ./scripts/cvphr_train.sh


set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bearinguav_mplconfig}"
export BEARING_UAV_RSI_TYPE="${BEARING_UAV_RSI_TYPE:-254k}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$MPLCONFIGDIR"

# ====================== parameters ======================
# Debug:    1.'37bc'+1+1+1; 2. 96+0/1+1+1. 
# Training: 1. 96+0/1+100+(1~100). 

rsi_id="${RSI_ID:-37bc}"       # 37bc for quick debug, 96 for full multi-city training
block="${N_BLOCK:-15}"
sample="${N_SAMPLE:-1}"        # 1 for debug, 100 for full paper baseline
epoch="${EPOCH:-1}"            # 1 for debug, 100 for full paper baseline
is_3d="${IS_3D:-1}"            # 0=2d: satellite view, 1=3d: U-S cross-view
factor_bslr="${FACTOR_BSLR:-0.5}"
model_class="${MODEL_CLASS:-PARCASGM_v5a}"
dataset_class="${DATASET_CLASS:-RSBlockDatasetPA_v3q}"
flag_test="${FLAG_TEST:-1}"
device_id="${DEVICE_ID:-0}"
gcth="${GCTH:-none}"

log_gcth="${gcth^^}"          # NONE
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/log/c4ma"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/train_${model_class}_d${rsi_id}_s${sample}_3d${is_3d}_e${epoch}_${timestamp}.log"

echo "[Run] repo_root=$REPO_ROOT"
echo "[Run] rsi_type=$BEARING_UAV_RSI_TYPE split=${BEARING_UAV_SPLIT_RATIO:-default}"
echo "[Run] log_file=$log_file"

nohup /usr/bin/time -v "$PYTHON_BIN" -m cvphr.train.cvphr_train \
    --is_3d "$is_3d" \
    --rsi_id "$rsi_id" \
    --n_sample "$sample" \
    --n_block "$block" \
    --num_epochs "$epoch" \
    --factor_bslr "$factor_bslr" \
    --model_class "$model_class" \
    --dataset_class "$dataset_class" \
    --flag_test "$flag_test" \
    --device_id "$device_id" \
    --gcth "$gcth" \
    > "$log_file" 2>&1 &

echo "Started! PID=$!"
echo "Log: $log_file"
