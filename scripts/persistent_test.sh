#!/bin/bash
# Independent PersistentBearing test entrypoint.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
DEVICE_ID="${DEVICE_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOPK="${TOPK:-5}"
FOREGROUND="${FOREGROUND:-0}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: bash scripts/persistent_test.sh /path/to/best_model.pth" >&2
    exit 2
fi

command=(
    "$PYTHON_BIN" -m cvphr.test.persistent_test
    --checkpoint "$CHECKPOINT"
    --device-id "$DEVICE_ID"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --topk "$TOPK"
)
if [ -n "${DATASET_DIR:-}" ]; then
    command+=(--dataset-dir "$DATASET_DIR")
fi
if [ "${MAX_BATCHES:-0}" -gt 0 ]; then
    command+=(--max-batches "$MAX_BATCHES")
fi

mkdir -p "$REPO_ROOT/log/persistent_bearing"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$REPO_ROOT/log/persistent_bearing/test_${timestamp}.log"
echo "[PersistentBearing] checkpoint=$CHECKPOINT"
echo "[PersistentBearing] log=$log_file"

if [ "$FOREGROUND" = "1" ]; then
    "${command[@]}" 2>&1 | tee "$log_file"
    exit 0
fi

nohup /usr/bin/time -v "${command[@]}" > "$log_file" 2>&1 &
echo "Started PID=$!"
