#!/bin/bash
# Launch mini-sglang-pearl server with nano-PEARL KV cache

# Cleanup shared memory on exit
cleanup() {
    rm -f /dev/shm/draft_group /dev/shm/target_group
}
trap cleanup EXIT

export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m minisgl_pearl.server \
    --model /home/ubuntu/models/meta-llama/Llama-3.1-70B-Instruct \
    --draft-model /home/ubuntu/models/meta-llama/Llama-3.2-3B-Instruct \
    --draft-tp 1 \
    --target-tp 2 \
    --port 30000 \
    --host 0.0.0.0 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 
    #--benchmark  # Enable speculative decoding benchmark (optional, comment out to skip)

