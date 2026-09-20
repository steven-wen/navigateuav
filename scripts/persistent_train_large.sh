#!/bin/bash
# PersistentBearing-L: large frozen foundation model with geometry-preserving MLPs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BACKBONE="${BACKBONE:-dinov3_convnext_base}"
export FEATURE_DIM="${FEATURE_DIM:-256}"
export ADAPTER_MLP_DEPTH="${ADAPTER_MLP_DEPTH:-3}"
export ADAPTER_MLP_RATIO="${ADAPTER_MLP_RATIO:-4}"
export SHARED_MLP_DEPTH="${SHARED_MLP_DEPTH:-3}"
export SHARED_MLP_RATIO="${SHARED_MLP_RATIO:-4}"
export MLP_DROPOUT="${MLP_DROPOUT:-0.05}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-20}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-5}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-0.0001}"

exec bash "$SCRIPT_DIR/persistent_train.sh"
