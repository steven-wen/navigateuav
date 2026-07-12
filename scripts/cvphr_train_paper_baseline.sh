#!/bin/bash
# Paper baseline training entrypoint for Bearing-UAV-90K cross-view setting.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

exec "$SCRIPT_DIR/cvphr_train.sh"
