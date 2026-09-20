#!/bin/bash
# Independent PersistentBearing training entrypoint. Baseline scripts are untouched.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/../Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/persistent_bearing}"
DEVICE_ID="${DEVICE_ID:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BACKBONE="${BACKBONE:-dinov3_convnext_small}"
FEATURE_DIM="${FEATURE_DIM:-64}"
ANGLE_BINS="${ANGLE_BINS:-36}"
TOPK="${TOPK:-5}"
ROTATION_SIGN="${ROTATION_SIGN:-1}"
ADAPTER_MLP_DEPTH="${ADAPTER_MLP_DEPTH:-0}"
ADAPTER_MLP_RATIO="${ADAPTER_MLP_RATIO:-4}"
SHARED_MLP_DEPTH="${SHARED_MLP_DEPTH:-0}"
SHARED_MLP_RATIO="${SHARED_MLP_RATIO:-4}"
MLP_DROPOUT="${MLP_DROPOUT:-0}"
HEADING_DISTRIBUTION="${HEADING_DISTRIBUTION:-legacy}"
HEADING_HIDDEN_DIM="${HEADING_HIDDEN_DIM:-64}"
HEADING_TOP_MODES="${HEADING_TOP_MODES:-3}"
HEADING_INITIAL_KAPPA="${HEADING_INITIAL_KAPPA:-20}"
HEADING_MIXTURE_WEIGHT="${HEADING_MIXTURE_WEIGHT:-0.5}"
CIRCULAR_WEIGHT="${CIRCULAR_WEIGHT:-0.1}"
UNIT_CIRCLE_WEIGHT="${UNIT_CIRCLE_WEIGHT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
SPLIT_RATIO="${SPLIT_RATIO:-0.7,0.2,0.1}"
FOREGROUND="${FOREGROUND:-0}"
RESUME="${RESUME:-}"

command=(
    "$PYTHON_BIN" -m cvphr.train.persistent_train
    --dataset-dir "$DATASET_DIR"
    --output-root "$OUTPUT_ROOT"
    --device-id "$DEVICE_ID"
    --backbone "$BACKBONE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --num-workers "$NUM_WORKERS"
    --feature-dim "$FEATURE_DIM"
    --angle-bins "$ANGLE_BINS"
    --topk "$TOPK"
    --rotation-sign "$ROTATION_SIGN"
    --adapter-mlp-depth "$ADAPTER_MLP_DEPTH"
    --adapter-mlp-ratio "$ADAPTER_MLP_RATIO"
    --shared-mlp-depth "$SHARED_MLP_DEPTH"
    --shared-mlp-ratio "$SHARED_MLP_RATIO"
    --mlp-dropout "$MLP_DROPOUT"
    --heading-distribution "$HEADING_DISTRIBUTION"
    --heading-hidden-dim "$HEADING_HIDDEN_DIM"
    --heading-top-modes "$HEADING_TOP_MODES"
    --heading-initial-kappa "$HEADING_INITIAL_KAPPA"
    --heading-mixture-weight "$HEADING_MIXTURE_WEIGHT"
    --circular-weight "$CIRCULAR_WEIGHT"
    --unit-circle-weight "$UNIT_CIRCLE_WEIGHT"
    --learning-rate "$LEARNING_RATE"
    --split-ratio "$SPLIT_RATIO"
)

if [ -n "${DINOV3_WEIGHTS:-}" ]; then
    command+=(--weights "$DINOV3_WEIGHTS")
fi
if [ -n "$RESUME" ]; then
    command+=(--resume "$RESUME")
fi
if [ "${MAX_TRAIN_BATCHES:-0}" -gt 0 ]; then
    command+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [ "${MAX_VAL_BATCHES:-0}" -gt 0 ]; then
    command+=(--max-val-batches "$MAX_VAL_BATCHES")
fi

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/log/persistent_bearing"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$REPO_ROOT/log/persistent_bearing/train_${timestamp}.log"

echo "[PersistentBearing] dataset=$DATASET_DIR"
echo "[PersistentBearing] backbone=$BACKBONE dim=$FEATURE_DIM adapter_mlp=$ADAPTER_MLP_DEPTH shared_mlp=$SHARED_MLP_DEPTH"
echo "[PersistentBearing] device=cuda:$DEVICE_ID micro_batch=$BATCH_SIZE accumulation=$GRAD_ACCUM_STEPS effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS)) angles=$ANGLE_BINS"
echo "[PersistentBearing] heading=$HEADING_DISTRIBUTION modes=$HEADING_TOP_MODES mixture_w=$HEADING_MIXTURE_WEIGHT circular_w=$CIRCULAR_WEIGHT unit_w=$UNIT_CIRCLE_WEIGHT"
if [ -n "$RESUME" ]; then
    echo "[PersistentBearing] resume=$RESUME"
fi
echo "[PersistentBearing] log=$log_file"

if [ "$FOREGROUND" = "1" ]; then
    "${command[@]}" 2>&1 | tee "$log_file"
    exit 0
fi

nohup /usr/bin/time -v "${command[@]}" > "$log_file" 2>&1 &
echo "Started PID=$!"
