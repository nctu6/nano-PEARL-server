# nano-PEARL-server

## Deployment Options

### Option 1: Mini-SGLang-PEARL Server (Recommended)
**Features**: Continuous batching, low TTFT, OpenAI-compatible API

#### Run in Docker
```bash
docker compose -f docker-compose-minisgl.yml up
```

#### Run on host machine
```bash
# 1-3: Same setup as Option 2
bash pearls.sh
```

**API Endpoint**: `http://localhost:30000/v1/completions`

---

### Option 2: Original FastChat PEARL Worker
**Features**: Basic PEARL speculative decoding

#### Run in Docker container
```bash
docker compose up
```

#### Run on host machine
0. create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```  
1. install FastChat
```bash
cd FastChat
pip install -e .
```
2. install nano-PEARL
```bash
cd nano-PEARL
pip install -e .
```
3. start controller
```bash
bash controller.sh
```
4. start worker
```bash
bash worker.sh
```
5. start api server
```bash
bash api.sh
```

---

## Architecture Comparison

| Feature | FastChat PEARL | Mini-SGLang-PEARL |
|---------|----------------|-------------------|
| TTFT | ~20s | <200ms ✅ |
| Continuous Batching | ❌ | ✅ |
| KV Cache | Basic | nano-PEARL native |
| API | FastChat | OpenAI-compatible |
| Script | `worker.sh` | `pearls.sh` |

## See Also

- [mini-sglang-pearl/README.md](mini-sglang-pearl/README.md) - Integration details
