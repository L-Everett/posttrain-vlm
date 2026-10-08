#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
VENV=/root/autodl-tmp/venv-vllm
MODEL=/root/autodl-tmp/merged/dpo
mkdir -p logs
nohup "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name chartqa-vl \
  --host 127.0.0.1 \
  --port 8000 \
  --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 589824}' \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > logs/vllm_serve.log 2>&1 &
echo "vLLM starting, pid $!"
echo "log: logs/vllm_serve.log"