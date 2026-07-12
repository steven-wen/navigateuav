#!/bin/bash
# Paper baseline training entrypoint for Bearing-UAV-90K cross-view setting.
# Usage: bash scripts/cvphr_train_paper_baseline.sh [GPU_ID]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

device_id="${1:-${DEVICE_ID:-4}}"
if ! [[ "$device_id" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU_ID must be a non-negative integer, got: $device_id" >&2
    exit 2
fi

export BEARING_UAV_RSI_TYPE="${BEARING_UAV_RSI_TYPE:-254k}"
export BEARING_UAV_SPLIT_RATIO="${BEARING_UAV_SPLIT_RATIO:-0.7,0.2,0.1}"
export MODEL_CLASS="${MODEL_CLASS:-PARCASGM_v5a}"
export DATASET_CLASS="${DATASET_CLASS:-RSBlockDatasetPA_v3q}"
export RSI_ID="${RSI_ID:-96}"
export N_BLOCK="${N_BLOCK:-15}"
export N_SAMPLE="${N_SAMPLE:-100}"
export IS_3D="${IS_3D:-1}"
export EPOCH="${EPOCH:-100}"
export FACTOR_BSLR="${FACTOR_BSLR:-0.5}"
export FLAG_TEST="${FLAG_TEST:-1}"
export DEVICE_ID="$device_id"

echo "[Run] gpu_device=cuda:$DEVICE_ID"

exec "$SCRIPT_DIR/cvphr_train.sh"
