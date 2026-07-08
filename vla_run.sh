#!/bin/bash
# VLA Run Wrapper — 自动设置 conda 环境并运行 vla_run.py
# Usage: ./vla_run.sh [args...]

DIR="$(cd "$(dirname "$0")" && pwd)"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vla_env
cd "$DIR"

# 禁用用户 site-packages 防止版本冲突
export PYTHONNOUSERSITE=1

exec python vla_run.py "$@"
