#!/bin/sh
cd "$(dirname "$0")" || exit 1
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs
python scripts/run_test.py "$@" 2>&1 | tee "logs/test_$(date +%Y%m%d_%H%M%S).log"