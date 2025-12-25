#!/usr/bin/env python3
"""
Concurrent streaming stress test for nano-PEARL/mini-sglang-pearl.

Runs batches at multiple concurrency levels, measures per-request TTFT,
token counts from usage chunks, and aggregates throughput metrics.
"""

import argparse
import asyncio
import aiohttp
import json
import random
import time
import math
import statistics
from dataclasses import dataclass
from typing import List, Optional

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
URL = "http://localhost:30000/v1/chat/completions"
MODEL = "pearl"
MAX_TOKENS = 256
PROMPTS = [
    "Hi",
    "Write a haiku about a coding bug.",
    "What are the benefits of distributed systems?",
    "Describe a futuristic city with flying cars.",
    "How does a neural network learn?",
    "Write a short story about a robot who loves gardening.",
    "Explain the difference between TCP and UDP.",
    "What is the meaning of life, the universe, and everything?",
    "Write a python function to calculate Fibonacci numbers.",
    "Describe the taste of a fresh strawberry."
]


@dataclass
class RequestStats:
    status: int
    start_time: float
    end_time: float
    first_token_time: Optional[float]
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time

    @property
    def itl(self) -> Optional[float]:
        if self.first_token_time is None or self.completion_tokens <= 0:
            return None
        return (self.end_time - self.first_token_time) / max(self.completion_tokens, 1)

    @property
    def throughput(self) -> float:
        dur = max(self.end_time - self.start_time, 1e-6)
        return self.completion_tokens / dur


async def wait_for_health():
    """Block until /health returns 200."""
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get("http://localhost:30000/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            print("⏳ Waiting for server health...")
            await asyncio.sleep(2)


async def send_stream_request(session: aiohttp.ClientSession, prompt: str, idx: int) -> RequestStats:
    """Send one streaming request and collect token counts/TTFT."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.time()
    first_token_time = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with session.post(URL, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return RequestStats(
                    status=resp.status,
                    start_time=start,
                    end_time=time.time(),
                    first_token_time=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    error=text.strip() or f"HTTP {resp.status}",
                )

            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line == b"data: [DONE]":
                        break
                    if not line.startswith(b"data: "):
                        continue
                    try:
                        data = json.loads(line[6:].decode("utf-8"))
                    except Exception:
                        continue

                    now = time.time()

                    # Usage chunk carries token counts
                    if "usage" in data:
                        usage = data["usage"] or {}
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        if first_token_time is None:
                            # Fallback: if the first meaningful chunk is usage, treat it as TTFT.
                            first_token_time = now
                        continue

                    # Regular delta chunk
                    if first_token_time is None:
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        # If any delta chunk arrives (even empty content), mark TTFT to avoid missing non-streaming cases.
                        if delta or delta.get("content") is not None:
                            first_token_time = now

            end = time.time()
            return RequestStats(
                status=resp.status,
                start_time=start,
                end_time=end,
                first_token_time=first_token_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=None,
            )
    except asyncio.TimeoutError:
        now = time.time()
        return RequestStats(
            status=408,
            start_time=start,
            end_time=now,
            first_token_time=first_token_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error="Timeout",
        )
    except Exception as e:
        now = time.time()
        return RequestStats(
            status=500,
            start_time=start,
            end_time=now,
            first_token_time=first_token_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=str(e),
        )


def summarize(concurrency: int, results: List[RequestStats], wall_time: float, iterations: int) -> str:
    def percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        ordered = sorted(values)
        k = (len(ordered) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return ordered[int(k)]
        return ordered[f] * (c - k) + ordered[c] * (k - f)

    def stats(values: List[float]):
        if not values:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            percentile(values, 0.50),
            percentile(values, 0.95),
            percentile(values, 0.99),
            statistics.pstdev(values),
        )

    successes = [r for r in results if r.status == 200]
    failures = len(results) - len(successes)

    total_output_tokens = sum(r.completion_tokens for r in successes)
    total_input_tokens = sum(r.prompt_tokens for r in successes)
    suc_count = len(successes) or 1
    avg_out_per_req = total_output_tokens / suc_count
    avg_in_per_req = total_input_tokens / suc_count

    ttft_values = [r.ttft for r in successes if r.ttft is not None]
    ttft_p50, ttft_p95, ttft_p99, ttft_std = stats(ttft_values)

    latency_values = [r.latency for r in successes]
    lat_p50, lat_p95, lat_p99, lat_std = stats(latency_values)

    tp_values = [r.throughput for r in successes]
    tp_p50, tp_p95, tp_p99, tp_std = stats(tp_values)

    rps = len(successes) / wall_time if wall_time > 0 else 0.0
    rpm = rps * 60
    # Per-user view: average throughput across requests
    avg_req_tp = sum(tp_values) / len(tp_values) if tp_values else 0.0
    # Aggregate view: tokens over wall time
    total_throughput = total_output_tokens / wall_time if wall_time > 0 else 0.0

    return (
        f"{concurrency:>5} | "
        f"{len(successes):>4}/{len(results):<4} | "
        f"{total_output_tokens:>10} | "
        f"{wall_time:>7.2f}s | "
        f"{total_throughput:>8.2f} tok/s (agg) | "
        f"{avg_req_tp:>8.2f} tok/s/req avg | "
        f"{tp_p50:>7.2f}/{tp_p95:>7.2f}/{tp_p99:>7.2f} tok/s req p50/p95/p99 | "
        f"{tp_std:>7.2f} tp std | "
        f"{avg_out_per_req:>7.1f} avg_out | "
        f"{avg_in_per_req:>7.1f} avg_in | "
        f"{ttft_p50*1000:>7.1f}/{ttft_p95*1000:>7.1f}/{ttft_p99*1000:>7.1f} ms TTFT p50/p95/p99 | "
        f"{ttft_std*1000:>7.1f} ms TTFT std | "
        f"{lat_p50*1000:>7.1f}/{lat_p95*1000:>7.1f}/{lat_p99*1000:>7.1f} ms latency p50/p95/p99 | "
        f"{lat_std*1000:>7.1f} ms latency std | "
        f"{rps:>6.2f} RPS | "
        f"{rpm:>7.1f} RPM | "
        f"fail={failures} | iters={iterations}"
    )


async def run_level(concurrency: int, iterations: int) -> str:
    """Run fixed-concurrency pool; refill as soon as a request completes.
    Total requests = iterations * concurrency (for compatibility with prior default).
    """
    all_results: List[RequestStats] = []
    total_requests = iterations * concurrency
    started = 0
    start_lock = asyncio.Lock()

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        async def worker(idx: int):
            nonlocal started
            while True:
                async with start_lock:
                    if started >= total_requests:
                        return
                    started += 1
                prompt = random.choice(PROMPTS)
                async with sem:
                    stats = await send_stream_request(session, prompt, idx)
                all_results.append(stats)

        pool_start = time.time()
        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        await asyncio.gather(*workers)
        pool_end = time.time()

    total_wall = pool_end - pool_start
    return summarize(concurrency, all_results, total_wall, iterations)


async def main():
    global URL, MODEL, MAX_TOKENS
    parser = argparse.ArgumentParser(description="Concurrent streaming stress test")
    parser.add_argument("--url", default=URL, help="Completions endpoint")
    parser.add_argument("--model", default=MODEL, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Generation max_tokens")
    parser.add_argument("--levels", default=",".join(str(x) for x in CONCURRENCY_LEVELS), help="Comma-separated concurrency levels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt sampling")
    parser.add_argument("--iterations", type=int, default=3, help="Number of waves per concurrency level")
    args = parser.parse_args()

    random.seed(args.seed)
    URL = args.url
    MODEL = args.model
    MAX_TOKENS = args.max_tokens
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    iterations = max(1, args.iterations)

    await wait_for_health()

    header = (
        " conc | succ  | total_tok |  time   |  total_tp |  avg_tp |    tp p50/p95/p99       |  tp std | avg_out | avg_in |        TTFT p50/p95/p99 (ms)        |  TTFT std |      Lat p50/p95/p99 (ms)       | Lat std |   RPS |    RPM  | fail | iters"
    )
    print(header)
    print("-" * len(header))

    for level in levels:
        line = await run_level(level, iterations)
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
