#!/bin/bash
# V1 dense frontend -> continuous correspondence -> iterative SE(2) refinement.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

DEFAULT_PYTHON="/home/zbw/miniconda3/envs/bearing/bin/python"
if [ ! -x "$DEFAULT_PYTHON" ]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/../Bearing_UAV_90K/c4m_254k_96bc_b15_s100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/continuous_se2_2d}"
DEVICE_ID="${DEVICE_ID:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-20}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ALIGNMENT_LR="${ALIGNMENT_LR:-0.0002}"
FRONTEND_LR="${FRONTEND_LR:-0.00005}"
BACKBONE_LR="${BACKBONE_LR:-0.000005}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.5}"
MATCH_DIM="${MATCH_DIM:-128}"
REFINEMENT_DIM="${REFINEMENT_DIM:-128}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-3}"
AMP_INIT_SCALE="${AMP_INIT_SCALE:-1024}"
DEFAULT_V1_CHECKPOINT="$REPO_ROOT/results/persistent_bearing_sat/persistent_dinov3_convnext_base_a36_d256_am3_sm3_b20x5_20260918_175726/best_model.pth"
V1_CHECKPOINT="${V1_CHECKPOINT-$DEFAULT_V1_CHECKPOINT}"
RESUME="${RESUME:-}"
FOREGROUND="${FOREGROUND:-0}"

command=(
    "$PYTHON_BIN" -m cvphr.train.continuous_se2_train
    --dataset-dir "$DATASET_DIR"
    --output-root "$OUTPUT_ROOT"
    --device-id "$DEVICE_ID"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --num-workers "$NUM_WORKERS"
    --alignment-lr "$ALIGNMENT_LR"
    --frontend-lr "$FRONTEND_LR"
    --backbone-lr "$BACKBONE_LR"
    --max-grad-norm "$MAX_GRAD_NORM"
    --match-dim "$MATCH_DIM"
    --refinement-dim "$REFINEMENT_DIM"
    --refinement-steps "$REFINEMENT_STEPS"
    --amp-init-scale "$AMP_INIT_SCALE"
)
if [ -n "${DINOV3_WEIGHTS:-}" ]; then
    command+=(--weights "$DINOV3_WEIGHTS")
fi
if [ -n "$V1_CHECKPOINT" ]; then
    command+=(--v1-checkpoint "$V1_CHECKPOINT")
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
if [ "${UNFREEZE_BACKBONE:-0}" = "1" ]; then
    command+=(--unfreeze-backbone)
fi

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/log/continuous_se2"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$REPO_ROOT/log/continuous_se2/train_2d_${timestamp}.log"

echo "[Continuous-SE2] dataset=$DATASET_DIR"
echo "[Continuous-SE2] V1_frontend=${V1_CHECKPOINT:-pretrained-DINO initialization}"
echo "[Continuous-SE2] micro_batch=$BATCH_SIZE accumulation=$GRAD_ACCUM_STEPS effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "[Continuous-SE2] alignment_lr=$ALIGNMENT_LR frontend_lr=$FRONTEND_LR backbone_lr=$BACKBONE_LR"
echo "[Continuous-SE2] log=$log_file"

if [ "$FOREGROUND" = "1" ]; then
    "${command[@]}" 2>&1 | tee "$log_file"
    exit 0
fi

nohup /usr/bin/time -v "${command[@]}" > "$log_file" 2>&1 &
echo "Started PID=$!"
