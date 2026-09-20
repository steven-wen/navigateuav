#!/bin/bash
# Train the DINOv3-Base PersistentBearing model on satellite-view queries.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET_DIR="$REPO_ROOT/../Bearing_UAV_90K/c4m_254k_96bc_b15_s100"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/persistent_bearing_sat}"

export BACKBONE="dinov3_convnext_base"
export FEATURE_DIM="256"
export ANGLE_BINS="36"
export TOPK="5"
export ROTATION_SIGN="1"
export ADAPTER_MLP_DEPTH="3"
export ADAPTER_MLP_RATIO="4"
export SHARED_MLP_DEPTH="3"
export SHARED_MLP_RATIO="4"
export MLP_DROPOUT="0.05"

export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-20}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-5}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-0.0001}"

exec bash "$SCRIPT_DIR/persistent_train.sh"
