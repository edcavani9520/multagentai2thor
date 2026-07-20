#!/usr/bin/env bash
set -euo pipefail

# ROS exports Python and shared-library paths that can override Conda's newer
# libstdc++ and break Transformers -> sklearn -> pyarrow imports.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_PREFIX:-$HOME/anaconda3}"
PYTHON_BIN="${RELAY_PYTHON:-$CONDA_HOME/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Relay Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$PYTHON_BIN" "$ROOT_DIR/relay_task_server.py" "$@"
