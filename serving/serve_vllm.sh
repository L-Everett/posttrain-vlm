#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
VENV="${VENV:-path/to/your/venv-vllm}"
MODEL="${MODEL:-path/to/your/merged/dpo}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_PIXELS="${MAX_PIXELS:-589824}"
NAME="${NAME:-chartqa-vl}"
mkdir -p logs
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > logs/vllm_serve.log 2>&1 &
echo "vLLM starting, pid $!"
echo "log: logs/vllm_serve.log"