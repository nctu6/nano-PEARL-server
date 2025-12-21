#!/bin/bash
# Launch mini-sglang-pearl server with nano-PEARL KV cache

python3 -m minisgl_pearl.server \
    --model /home/ubuntu/models/Qwen/Qwen3-32B \
    --draft-model /home/ubuntu/models/Qwen/Qwen3-1.7B \
    --tp 2 \
    --port 30000 \
    --host 0.0.0.0 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384
