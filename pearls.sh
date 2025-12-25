#!/bin/bash
# Launch mini-sglang-pearl server with nano-PEARL KV cache

# Cleanup shared memory on exit
cleanup() {
    rm -f /dev/shm/draft_group /dev/shm/target_group
}
trap cleanup EXIT

# Check if .venv exists locally
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

# Configurable defaults (override via env)
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/models/Qwen/Qwen3-32B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/home/ubuntu/models/Qwen/Qwen3-1.7B}
DRAFT_TP=${DRAFT_TP:-1}
TARGET_TP=${TARGET_TP:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}

# Optional benchmark flag (set ENABLE_BENCHMARK=1 to enable)
BENCH_FLAG=""
if [ -n "${ENABLE_BENCHMARK:-}" ]; then
    BENCH_FLAG="--benchmark"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

$PYTHON -m minisgl_pearl.server \
    --model "$MODEL_PATH" \
    --draft-model "$DRAFT_MODEL_PATH" \
    --draft-tp "$DRAFT_TP" \
    --target-tp "$TARGET_TP" \
    --port "$PORT" \
    --host "$HOST" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    $BENCH_FLAG  # Enable speculative decoding benchmark by setting ENABLE_BENCHMARK=1
