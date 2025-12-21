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

$PYTHON -m minisgl_pearl.server \
    --model /home/ubuntu/models/Qwen/Qwen3-32B \
    --draft-model /home/ubuntu/models/Qwen/Qwen3-1.7B \
    --draft-tp 1 \
    --target-tp 1 \
    --port 30000 \
    --host 0.0.0.0 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384
