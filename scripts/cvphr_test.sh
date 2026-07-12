#!/bin/bash
# Function：Bearing-UAV (cvphr) model test.
# Operation：
#   cd /your/path/of/proj/bearinguav
#   chmod +x ./scripts/cvphr_test.sh
#   ./scripts/cvphr_test.sh


set -e
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
echo "[Run] project_root=$project_root"
export PYTHONPATH="$project_root:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bearinguav_mplconfig}"
export BEARING_UAV_RSI_TYPE="${BEARING_UAV_RSI_TYPE:-254k}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$MPLCONFIGDIR"

# ====================== parameters ======================
# Debug: '37bc'/96+1+3D model .
# Test:  96 + 100 + 2D/3D/User model.

rsi_id="${RSI_ID:-37bc}"       # 37bc for debug; 96 for full multi-city test.
sample="${N_SAMPLE:-1}"        # 1 for debug, 100 for full paper baseline.
is_3d="${IS_3D:-1}"            # 0=2D: satellite view; 1=3D: UAV view.
model_class="${MODEL_CLASS:-PARCASGM_v5a}"
dataset_class="${DATASET_CLASS:-RSBlockDatasetPA_v3q}"
device_id="${DEVICE_ID:-0}"

# Model weight path: 1.2D; 2. 3D; 3. Your trained model.
# 2D model: Pre-trained cvphr_2d_best_model_dir 
# bestpth_dir="./Bearing_UAV/satellite_view" 

# 3D model: Pre-trained cvphr_3d_best_model_dir 
bestpth_dir="${BESTPTH_DIR:-./Bearing_UAV/cross_view}"

# User model: Your trained model dir:
# bestpth_dir="${project_root}/results/c4ma/phr5_~~~"

# ====================== log ======================
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${project_root}/log/c4ma"
mkdir -p "$LOG_DIR"

log_file="${LOG_DIR}/test_d${rsi_id}_3d${is_3d}_${timestamp}.log"

echo "[Run] log_file=$log_file"
echo "[Run] rsi_type=$BEARING_UAV_RSI_TYPE bestpth_dir=$bestpth_dir"

if [ ! -f "$bestpth_dir/best_model.pth" ]; then
    echo "Error: cannot find $bestpth_dir/best_model.pth"
    echo "Set BESTPTH_DIR to a trained result directory or download the official Bearing_UAV weights."
    exit 1
fi

# ====================== exe ======================
nohup /usr/bin/time -v "$PYTHON_BIN" -m cvphr.test.cvphr_test \
    --rsi_id "$rsi_id" \
    --n_sample "$sample" \
    --is_3d "$is_3d" \
    --model_class "$model_class" \
    --dataset_class "$dataset_class" \
    --device_id "$device_id" \
    --bestpth_dir "$bestpth_dir" \
    > "$log_file" 2>&1 &

echo "Started! PID=$!"
echo "Log: $log_file"
