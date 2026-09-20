#!/bin/bash
# PersistentBearing closed-loop navigation entrypoint.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
RSI_ID="${RSI_ID:-37bc}"
TRAJ_ID="${TRAJ_ID:-50}"
UAV_STEP="${UAV_STEP:-25}"
ARRIVAL_THRESHOLD="${ARRIVAL_THRESHOLD:-20}"
DEVICE_ID="${DEVICE_ID:-0}"
FOREGROUND="${FOREGROUND:-0}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: bash scripts/persistent_nav.sh /path/to/best_model.pth" >&2
    exit 2
fi

command=(
    "$PYTHON_BIN" -m naver.runners.persistent_nav
    --checkpoint "$CHECKPOINT"
    --rsi-id "$RSI_ID"
    --traj-id "$TRAJ_ID"
    --uav-step "$UAV_STEP"
    --arrival-threshold "$ARRIVAL_THRESHOLD"
    --device-id "$DEVICE_ID"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
    command+=(--dry-run)
fi

mkdir -p "$REPO_ROOT/log/persistent_bearing"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$REPO_ROOT/log/persistent_bearing/nav_${RSI_ID}_${TRAJ_ID}_${timestamp}.log"
echo "[PersistentBearing] checkpoint=$CHECKPOINT"
echo "[PersistentBearing] route=${RSI_ID}_${TRAJ_ID} log=$log_file"

if [ "$FOREGROUND" = "1" ]; then
    "${command[@]}" 2>&1 | tee "$log_file"
    exit 0
fi

nohup /usr/bin/time -v "${command[@]}" > "$log_file" 2>&1 &
echo "Started PID=$!"
