# Mini-SGLang-PEARL Integration

Combines mini-sglang's continuous batching infrastructure with nano-PEARL's speculative decoding engine.

## Architecture

```
┌─────────────────────────────────────┐
│   OpenAI-Compatible API Server      │
│   (FastAPI + continuous batching)   │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│        PEARL Engine Wrapper         │
│  - Request scheduling (no delays!)  │
│  - nano-PEARL KV cache management   │
└──────────────┬──────────────────────┘
               ↓
       ┌───────┴───────┐
       ↓               ↓
┌─────────────┐ ┌─────────────┐
│ Draft Model │ │Target Model │
│ (nano-PEARL)│ │(nano-PEARL) │
└─────────────┘ └─────────────┘
```

## Key Features

✅ **Continuous Batching**: No BATCH_MAX_WAIT delays, immediate processing
✅ **nano-PEARL KV Cache**: Uses nano-PEARL's native KV cache architecture (NOT mini-sglang's page cache)  
✅ **Speculative Decoding**: PEARL's draft + verify approach  
✅ **OpenAI API**: Compatible with OpenAI client libraries  
✅ **Preserves Original**: Doesn't modify FastChat PEARL worker

## Installation

```bash
cd /path/to/nano-PEARL-server
pip install -e mini-sglang-pearl
```

## Usage

### Launch Server

```bash
bash pearls.sh
```

Or manually:

```bash
python -m minisgl_pearl.server \
    --model /path/to/Qwen3-32B \
    --draft-model /path/to/Qwen3-1.7B \
    --tp 2 \
    --port 30000
```

### Send Requests

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### Streaming

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a story",
    "max_tokens": 256,
    "temperature": 0.8,
    "stream": true
  }'
```

## Performance

### Expected Improvements vs FastChat PEARL

| Metric | FastChat PEARL | Mini-SGLang-PEARL |
|--------|----------------|-------------------|
| TTFT (single request) | ~20s | <200ms |
| TTFT (concurrent) | ~20s | <500ms |
| Throughput | Limited | 2-3x higher |
| Concurrent capacity | 32 (hard limit) | Dynamic batching |

## Architecture Details

### nano-PEARL KV Cache

This integration uses **nano-PEARL's native KV cache**, not mini-sglang's page cache or Radix cache.

Advantages:
- Designed specifically for speculative decoding
- Optimized for draft + target model coordination  
- Proven stable in production

### Continuous Batching

Requests are processed immediately upon arrival, no waiting for batch to fill:

```python
# Old FastChat approach (1s delay)
while elapsed < BATCH_MAX_WAIT:  # 1.0 second wait!
    await asyncio.sleep(0.01)

# New approach (immediate)
while pending_requests:
    batch = schedule_next_batch()  # No delay!
    await generate(batch)
```

## Comparison: Original vs Integrated

| Component | Original (FastChat) | This Integration |
|-----------|---------------------|------------------|
| API Server | FastChat | mini-sglang (OpenAI compatible) |
| Scheduling | Simple batch | Continuous batching |
| KV Cache | Basic | **nano-PEARL native** |
| PEARL Engine | ✅ Preserved | ✅ Used |
| Worker Script | worker.sh | pearls.sh |
| Status | Untouched | New implementation |

## Development

### Project Structure

```
mini-sglang-pearl/
├── minisgl_pearl/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   └── pearl_engine.py      # PEARL engine wrapper
│   └── server/
│       ├── __init__.py
│       └── __main__.py           # API server
├── pyproject.toml
└── README.md
```

### Testing

```bash
# Health check
curl http://localhost:30000/health

# List models
curl http://localhost:30000/v1/models
```

## Future Enhancements

- [ ] True tokenizer integration (currently placeholder)
- [ ] Streaming improvements (real incremental tokens)
- [ ] Metrics and monitoring
- [ ] Batch size optimization
- [ ] Multi-GPU load balancing

## License

Same as nano-PEARL and mini-sglang parent projects.
