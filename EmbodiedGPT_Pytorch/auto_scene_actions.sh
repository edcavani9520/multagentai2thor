#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
EMBODIED_ROOT="${EMBODIED_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-qwen35}"
SEND_ACTIONS_URL="${SEND_ACTIONS_URL:-http://127.0.0.1:19001/execute_actions}"
QWEN_MODEL="${QWEN_MODEL:-$BASE_DIR/models/Qwen3.5-4B}"
DEVICE="${DEVICE:-cuda}"
QWEN_DEVICE_MAP="${QWEN_DEVICE_MAP:-auto}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.2}"

cd "$EMBODIED_ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export HF_HOME="${HF_HOME:-$BASE_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python demo/auto_scene_actions.py \
  --execute-actions-url "$SEND_ACTIONS_URL" \
  --qwen-model "$QWEN_MODEL" \
  --device "$DEVICE" \
  --qwen-device-map "$QWEN_DEVICE_MAP" \
  --qwen-dtype "$QWEN_DTYPE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  "$@"
