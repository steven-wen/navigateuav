#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"
PYTHON_BIN="${PYTHON_BIN:-/home/zbw/miniconda3/envs/bearing/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to best_model.pth}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/../Bearing_UAV_90K/c4m_254k_96bc_b15_s100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE_ID="${DEVICE_ID:-0}"

command=(
    "$PYTHON_BIN" -m cvphr.test.continuous_se2_test
    --checkpoint "$CHECKPOINT"
    --dataset-dir "$DATASET_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device-id "$DEVICE_ID"
)
if [ "${MAX_BATCHES:-0}" -gt 0 ]; then
    command+=(--max-batches "$MAX_BATCHES")
fi
if [ -n "${OUTPUT_DIR:-}" ]; then
    command+=(--output-dir "$OUTPUT_DIR")
fi
exec "${command[@]}"
