# nano-PEARL-server

**Developed and Maintained by Team UnieAI**

A high-performance speculative decoding server optimized for high-concurrency LLM inference, featuring PEARL (Proximal Early Accept/Reject Logic) algorithm with continuous batching and dynamic gamma adjustment.

---

## 🗂️ Repository Layout & Service Flow
- `mini-sglang-pearl/`: FastAPI/OpenAI server + continuous batching scheduler; wraps `nano_pearl.PEARLEngine` (draft + target speculative decoding, native KV cache). Streaming uses單一 generator + per-request queue，避免併發搶 generator。
- `nano-PEARL/`: 核心 speculative 解碼引擎、CUDA/FlashAttention/Triton kernels、KV cache 管理。
- `mini-sglang/`: 連續 batching 基礎（vendored），API 相容層。
- Scripts: `pearls.sh`（啟動/清理 /dev/shm）、`stress_test.py`（併發階梯 TTFT/MAT/吞吐）、`monitor_streaming.py`（串流 TUI）、`benchmark_speculative.py`（Auto Gamma 後的 perf）。
- Logs: `server*.log`; Docker: `docker-compose*.yml`, `Dockerfile`.
- 監控：服務執行時每秒輸出 `📊 Streaming throughput`（tok/s，計入 acc token）與 MAT；批次完成時亦輸出總吞吐/MAT。

## 🎯 Project Overview

nano-PEARL-server is UnieAI's flagship project focused on **speculative decoding optimization for high-concurrency scenarios**. By combining draft-target model verification with intelligent batch scheduling, we achieve significant throughput improvements while maintaining output quality.

### Key Innovations

- **🚀 Dynamic Gamma Adjustment**: Automatically tunes speculation window based on batch size and workload
- **⚡ Continuous Batching**: Eliminates artificial batching delays for immediate request processing  
- **🎯 Speculative Decoding**: PEARL algorithm with draft model acceleration and target model verification
- **📊 Auto Set Gamma**: Profiling-based gamma optimization across different concurrency levels

---

## 🏗️ Architecture

### Detailed Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server (Port 30000)                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  OpenAI-Compatible Endpoints (/v1/completions, /v1/chat/completions)    │  │
│  │  - Parse JSON requests (prompt, temp, top_k, top_p, penalties, etc.)    │  │
│  │  - Stream/non-stream response handling                                  │  │
│  └────────────────────────────┬─────────────────────────────────────────────┘  │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────────────────┐
│                    PEARL Engine Wrapper (mini-sglang)                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Continuous Batching Scheduler                                           │  │
│  │  - No BATCH_MAX_WAIT delays (immediate processing)                       │  │
│  │  - Dynamic gamma selection based on batch size                           │  │
│  │  - Request queue: pending → running → finished                           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Parameter Conversion Layer                                              │  │
│  │  - mini-sglang SamplingParams → nano-PEARL SamplingParams                │  │
│  │  - Handles top_p, penalties as dynamic attributes                        │  │
│  └────────────────────────────┬─────────────────────────────────────────────┘  │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  ↓
                    ┌─────────────────────────────┐
                    │  IPC via Shared Memory      │
                    │  /dev/shm/draft_group       │
                    │  /dev/shm/target_group      │
                    │  Multiprocessing Events     │
                    └───────┬──────────┬──────────┘
                            ↓          ↓
          ┌─────────────────────────────────────────────────────┐
          ↓                                                     ↓
┌───────────────────────┐                           ┌───────────────────────┐
│   Draft Model Process │                           │  Target Model Process │
│   (GPU 0, Rank 0)     │                           │  (GPU 1, Rank 1)      │
├───────────────────────┤                           ├───────────────────────┤
│                       │                           │                       │
│  ┌─────────────────┐  │                           │  ┌─────────────────┐  │
│  │  Model Forward  │  │                           │  │  Model Forward  │  │
│  │  (Qwen3-1.7B)   │  │                           │  │  (Qwen3-32B)    │  │
│  └────────┬────────┘  │                           │  └────────┬────────┘  │
│           ↓           │                           │           ↓           │
│  ┌─────────────────┐  │    ┌──────────────────┐   │  ┌─────────────────┐  │
│  │ Attention Layer │  │    │  PEARL Verifier  │   │  │ Attention Layer │  │
│  │  - Flash Attn   │←─┼────┤                  │───┼─→│  - Flash Attn   │  │
│  │  - Triton       │  │    │  Draft → Target  │   │  │  - Triton       │  │
│  └────────┬────────┘  │    │  Token Matching  │   │  └────────┬────────┘  │
│           ↓           │    │  Accept/Reject   │   │           ↓           │
│  ┌─────────────────┐  │    │                  │   │  ┌─────────────────┐  │
│  │   KV Cache      │  │    │  MAT Tracking    │   │  │   KV Cache      │  │
│  │ ┌─────────────┐ │  │    └──────────────────┘   │  │ ┌─────────────┐ │  │
│  │ │ Block Mgr   │ │  │                           │  │ │ Block Mgr   │ │  │
│  │ │ 256-tok     │ │  │                           │  │ │ 256-tok     │ │  │
│  │ │ blocks      │ │  │                           │  │ │ blocks      │ │  │
│  │ └─────────────┘ │  │                           │  │ └─────────────┘ │  │
│  │  • Allocate     │  │                           │  │  • Allocate     │  │
│  │  • Free         │  │                           │  │  • Free         │  │
│  │  • Copy (fork)  │  │                           │  │  • Copy (fork)  │  │
│  └─────────────────┘  │                           │  └─────────────────┘  │
│           ↓           │                           │           ↓           │
│  ┌─────────────────┐  │                           │  ┌─────────────────┐  │
│  │    Sampler      │  │                           │  │    Sampler      │  │
│  │  - Temperature  │  │                           │  │  - Temperature  │  │
│  │  - Top-k/Top-p  │  │                           │  │  - Top-k/Top-p  │  │
│  │  - Penalties    │  │                           │  │  - Penalties    │  │
│  └─────────────────┘  │                           │  └─────────────────┘  │
│           ↓           │                           │           ↓           │
│   Gamma tokens        │                           │   Verify + Sample     │
│   (speculative)       │                           │   (correct if reject) │
└───────────────────────┘                           └───────────────────────┘

PEARL Speculative Decoding Flow:
1. Draft generates gamma=9 candidate tokens in one forward pass
2. Target verifies all candidates in parallel (single forward pass)
3. Accept prefix of verified tokens (MAT ≈ 7-8 on average)
4. Reject and correct remainder from draft speculation
5. Return accepted tokens to client, continue generation

KV Cache Details:
• Block-based allocation (256 tokens/block)
• Shared memory for IPC between draft/target processes  
• Efficient fork/copy for draft speculation verification
• Dynamic allocation as sequences grow
• Freed immediately when sequences complete
```

### Core Technologies

#### Kernel Implementation
- **Triton**: Custom CUDA kernels for attention operations
- **Flash Attention 2**: Memory-efficient attention with varlen support
  - `flash_attn_varlen_func`: Variable-length prefill
  - `flash_attn_with_kvcache`: Decode with KV cache

#### PEARL Algorithm
1. **Draft Phase**: Small model generates `gamma` candidate tokens
2. **Verification Phase**: Large model validates candidates in parallel
3. **Accept/Reject**: Accept verified prefix, reject and correct remainder
4. **MAT (Mean Accepted Tokens)**: Average tokens accepted per step

#### KV Cache Management
- **Block-based allocation**: 256-token blocks with dynamic allocation
- **nano-PEARL native**: Purpose-built for speculative decoding (not vLLM/SGLang PagedAttention)
- **Memory efficient**: Shared cache between draft verification rounds

---

## 🤖 Supported Models

### Currently Supported
- **Llama** (Llama 2, Llama 3, etc.)
- **Qwen2** (Qwen2-0.5B to Qwen2-72B)
- **Qwen3** (Latest Qwen3 series)

### Model Requirements
- Draft model: Typically 1-4B parameters (e.g., Qwen3-1.7B)
- Target model: Larger model (e.g., Qwen3-32B, Llama 3-70B)
- **Recommendation**: ~10-20x speed ratio between draft and target for optimal gamma

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/UnieAI/nano-PEARL-server
cd nano-PEARL-server

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install nano-PEARL
cd nano-PEARL
pip install -e .

# Install mini-sglang-pearl
cd ../mini-sglang-pearl
pip install -e .
```

### Quick Start

```bash
# Edit pearls.sh to set your model paths
nano pearls.sh

# Launch server
bash pearls.sh
```

The server will:
1. Initialize draft and target models
2. Run Auto Set Gamma profiling (computes optimal gamma for batch sizes 1-256)
3. Start listening on `http://0.0.0.0:30000`

### Configuration

Edit `pearls.sh`:

```bash
--model /path/to/target/model          # Target model (e.g., Qwen3-32B)
--draft-model /path/to/draft/model     # Draft model (e.g., Qwen3-1.7B)
--draft-tp 1                           # Draft tensor parallel size
--target-tp 2                          # Target tensor parallel size
--port 30000                           # Server port
--max-num-seqs 512                     # Max concurrent sequences
--max-num-batched-tokens 16384         # Max tokens per batch
--gpu-memory-utilization 0.9           # GPU memory utilization (default 0.9)
```

### API Usage

**Standard OpenAI Format (Recommended)**

```bash
# Chat completion with messages array
curl --request POST \
  --url http://localhost:30000/v1/chat/completions \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pearl",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 50,
    "max_tokens": 256,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }'

# Streaming response
curl --request POST \
  --url http://localhost:30000/v1/chat/completions \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pearl",
    "messages": [
      {"role": "user", "content": "Write a poem about AI"}
    ],
    "max_tokens": 100,
    "temperature": 0.8,
    "stream": true
  }'
```

**Legacy Format (Also Supported)**

```bash
# Direct prompt (backward compatibility)
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
  }'
```

---

## 🔧 Debugging & Troubleshooting

### Common Issues

#### 1. NCCL Initialization Errors
```
DistNetworkError: address already in use: port 2333
```
**Solution**: Kill existing processes and clean shared memory
```bash
fuser -k 2333/tcp
fuser -k 30000/tcp
rm -f /dev/shm/draft_group /dev/shm/target_group
```

#### 2. KV Cache Allocation Warnings
```
WARNING: block_manager.can_allocate(seq): False
```
**Cause**: Insufficient GPU memory for large batch sizes
**Solution**: Normal during Auto Set Gamma profiling; adjust `--max-num-seqs` for production

#### 3. OOM During Auto Set Gamma
**Solution**: Reduce batch size range in `pearl_model_runner.py`:
```python
bs = [1, 2, 4, 8, 16, 32, 64]  # Instead of up to 256
```

### Logging

Monitor server output for key metrics:
```
batch size: 32, draft speed: 221.32 tok/s, target speed: 24.46 tok/s, gamma: 9
MAT: 7.2, throughput: 185.5 tok/s, per-request: 5.8 tok/s
```

### Profiling

Enable detailed logging:
```python
# In pearl_engine.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📊 Performance Characteristics

### Auto Set Gamma Results (Example with Qwen3-32B + Qwen3-1.7B)

| Batch Size | Draft Speed | Target Speed | Gamma | MAT  | Throughput | Per-Request |
|-----------|-------------|--------------|-------|------|------------|-------------|
| 1         | 254.7 tok/s | 25.8 tok/s   | 10    | 8.5  | 220 tok/s  | 220 tok/s   |
| 4         | 236.9 tok/s | 25.3 tok/s   | 9     | 7.8  | 810 tok/s  | 202 tok/s   |
| 16        | 228.7 tok/s | 24.9 tok/s   | 9     | 7.1  | 2.8k tok/s | 175 tok/s   |
| 64        | 199.5 tok/s | 23.4 tok/s   | 9     | 6.2  | 9.1k tok/s | 142 tok/s   |
| 128       | 159.4 tok/s | 21.6 tok/s   | 7     | 5.1  | 14k tok/s  | 109 tok/s   |

*MAT (Mean Accepted Tokens) = Average tokens accepted per speculative decoding step*

### Speedup Analysis
- **Single request**: ~8-10x speedup over target-only inference
- **High concurrency (64+)**: ~6-7x speedup with maintained throughput

---

## 🔮 Future Work

### Model Support Expansion
- **Mistral / Mixtral**: MoE architecture support
- **DeepSeek**: Extended context models
- **Phi-3**: Small language models
- **Gemma**: Google's open models

### PD (Prefill-Decode) Separation
- Separate prefill and decode kernels for better cache locality
- Optimized scheduling for mixed workloads
- Chunked prefill for long contexts

### Tool Use & Function Calling
- Structured output support
- Tool/function call detection and routing
- JSON mode with schema validation

### Cross-Platform Support
- **ROCm**: AMD GPU support
- **Metal**: Apple Silicon optimization
- **CPU**: Fallback for inference without GPUs
- **Multi-node**: Distributed inference across machines

### Advanced Features
- **Prefix caching**: Reuse common prompt prefixes
- **LoRA support**: Dynamic adapter loading
- **Quantization**: INT8/INT4 for memory efficiency
- **Custom samplers**: Top-p, top-k, temperature annealing

---

## 📄 API Reference

### Parameters

| Parameter            | Type    | Default | Description                               |
|---------------------|---------|---------|-------------------------------------------|
| `model`             | string  | "pearl" | Model identifier (for compatibility)      |
| `messages`          | array   | -       | Chat messages with role and content       |
| `prompt`            | string  | -       | Direct prompt text (legacy format)        |
| `max_tokens`        | integer | 256     | Maximum tokens to generate                |
| `temperature`       | float   | 1.0     | Sampling temperature (0.0 = greedy)       |
| `top_k`             | integer | -1      | Top-k filtering (-1 = disabled)           |
| `top_p`             | float   | 1.0     | Nucleus sampling (1.0 = disabled)         |
| `frequency_penalty` | float   | 0.0     | Penalize frequent tokens (0.0-2.0)        |
| `presence_penalty`  | float   | 0.0     | Penalize repeated tokens (0.0-2.0)        |
| `stream`            | boolean | false   | Enable streaming response                 |

### Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion (same handler)

---

## 🧪 Testing & Benchmarking


### Benchmark - Speculative Decoding Performance

Measures **actual PEARL speculation** with real MAT and speedup.

**Usage:**
```bash
python3 benchmark_speculative.py   # Run after server starts
```

**What It Measures:**

| Phase | Auto Set Gamma | Benchmark |
|-------|---------------|-----------|
| **Purpose** | Calculate gamma | Test real speculation |
| **Speculation** | ❌ No | ✅ Yes |
| **MAT** | 1.0 | ~7-8 |
| **Speedup** | Baseline | ~7x |
| **Duration** | ~16s | ~30-60s |

**Example Results:**
```
Batch    Gamma    Throughput      MAT      Speedup   
------------------------------------------------------
1        10       182.93 tok/s    7.32     7.32x
4        9        720.48 tok/s    7.20     7.20x
```

---

## 📝 Citation

If you use nano-PEARL-server in your research, please cite:

```bibtex
@software{nano_pearl_server,
  title={nano-PEARL-server: High-Concurrency Speculative Decoding with Dynamic Gamma},
  author={UnieAI Team},
  year={2024},
  url={https://github.com/UnieAI/nano-PEARL-server}
}
```

---

## 📜 License

[Specify license here]

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

**Team UnieAI**  
- GitHub: [@UnieAI](https://github.com/UnieAI)
- Email: [contact@unieai.com](mailto:contact@unieai.com)

---

**Built with ❤️ by Team UnieAI**
