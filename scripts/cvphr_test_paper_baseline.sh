#!/bin/bash
# Paper baseline testing entrypoint for official or trained cross-view weights.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BEARING_UAV_RSI_TYPE="${BEARING_UAV_RSI_TYPE:-254k}"
export BEARING_UAV_SPLIT_RATIO="${BEARING_UAV_SPLIT_RATIO:-0.7,0.2,0.1}"
export MODEL_CLASS="${MODEL_CLASS:-PARCASGM_v5a}"
export DATASET_CLASS="${DATASET_CLASS:-RSBlockDatasetPA_v3q}"
export RSI_ID="${RSI_ID:-96}"
export N_SAMPLE="${N_SAMPLE:-100}"
export IS_3D="${IS_3D:-1}"
export BESTPTH_DIR="${BESTPTH_DIR:-./Bearing_UAV/cross_view}"

exec "$SCRIPT_DIR/cvphr_test.sh"
