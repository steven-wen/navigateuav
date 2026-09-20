#!/bin/bash
# DINOv3-Base with symmetry-aware circular multimodal heading estimation.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HEADING_DISTRIBUTION="von_mises_mixture"
export HEADING_HIDDEN_DIM="${HEADING_HIDDEN_DIM:-64}"
export HEADING_TOP_MODES="${HEADING_TOP_MODES:-3}"
export HEADING_INITIAL_KAPPA="${HEADING_INITIAL_KAPPA:-20}"
export HEADING_MIXTURE_WEIGHT="${HEADING_MIXTURE_WEIGHT:-0.5}"
export CIRCULAR_WEIGHT="${CIRCULAR_WEIGHT:-0.1}"
export UNIT_CIRCLE_WEIGHT="${UNIT_CIRCLE_WEIGHT:-0.05}"

exec bash "$SCRIPT_DIR/persistent_train_large.sh"
