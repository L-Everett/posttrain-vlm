#!/bin/sh
cd "$(dirname "$0")" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p logs
python scripts/run_train.py "$@" 2>&1 | tee "logs/train_$(date +%Y%m%d_%H%M%S).log"