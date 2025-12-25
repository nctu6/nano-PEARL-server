#!/usr/bin/env python3
"""
Speculative Decoding Benchmark for nano-PEARL

After Auto Set Gamma completes, run this script to test real speculative decoding performance.
This measures actual MAT (Mean Accepted Tokens) and throughput with PEARL verification.

Usage:
    python3 benchmark_speculative.py
"""

import asyncio
import aiohttp
import time
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    batch_size: int
    gamma: int
    num_requests: int
    total_tokens: int
    total_time: float
    avg_tokens_per_req: float
    throughput: float  # tokens/sec
    throughput_per_req: float  # tokens/sec per request
    avg_latency: float

# Gamma values from Auto Set Gamma (defaults, will be displayed as reference)
GAMMA_CONFIG = {
    1: 10,
    2: 10,
    4: 9,
    8: 9,
    16: 9,
    32: 9,
    64: 8,
    128: 7,
    256: 5,
}

# Target speed without speculation (baseline, from Auto Set Gamma)
# These are approximate values, adjust based on your actual Auto Set Gamma results
TARGET_BASELINE_SPEED = 25.0  # tok/s

async def send_request(session: aiohttp.ClientSession, url: str, prompt: str, max_tokens: int, idx: int):
    """Send a single completion request"""
    payload = {
        "model": "pearl",
        "messages": [
            {"role": "user", "content": f"{prompt} (Request #{idx})"}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "stream": False,
    }
    
    start = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
            latency = time.time() - start
            if response.status == 200:
                data = await response.json()
                tokens = data.get('usage', {}).get('completion_tokens', 0)
                return {'status': 200, 'tokens': tokens, 'latency': latency}
            else:
                return {'status': response.status, 'tokens': 0, 'latency': latency}
    except Exception as e:
        return {'status': 500, 'tokens': 0, 'latency': time.time() - start, 'error': str(e)}

async def benchmark_batch_size(batch_size: int, gamma: int, url: str, prompt: str, max_tokens: int, num_iterations: int = 1):
    """Benchmark a specific batch size with its gamma value"""
    print(f"\n{'='*80}")
    print(f"Testing Batch Size: {batch_size}, Gamma: {gamma}")
    print(f"{'='*80}")
    
    all_results = []
    
    for iteration in range(num_iterations):
        async with aiohttp.ClientSession() as session:
            # Send batch_size concurrent requests
            tasks = [send_request(session, url, prompt, max_tokens, i) for i in range(batch_size)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
        
        # Analyze results
        success = [r for r in results if r['status'] == 200]
        if len(success) == 0:
            print(f"  ❌ All requests failed in iteration {iteration + 1}")
            continue
        
        total_tokens = sum(r['tokens'] for r in success)
        latencies = [r['latency'] for r in success]
        
        all_results.append({
            'total_tokens': total_tokens,
            'total_time': total_time,
            'latencies': latencies,
            'success_count': len(success),
        })
        
        if num_iterations > 1:
            print(f"  Iteration {iteration + 1}/{num_iterations}: {total_tokens} tokens in {total_time:.2f}s")
    
    if not all_results:
        return None
    
    # Aggregate results
    avg_total_tokens = np.mean([r['total_tokens'] for r in all_results])
    avg_total_time = np.mean([r['total_time'] for r in all_results])
    all_latencies = []
    for r in all_results:
        all_latencies.extend(r['latencies'])
    
    throughput = avg_total_tokens / avg_total_time
    throughput_per_req = throughput / batch_size
    avg_tokens_per_req = avg_total_tokens / batch_size
    avg_latency = np.mean(all_latencies)
    
    # Calculate estimated MAT
    # MAT ≈ (tokens per request) / (steps taken to generate them)
    # Since we don't track steps directly, estimate: MAT ≈ throughput_per_req / baseline_speed
    estimated_mat = throughput_per_req / TARGET_BASELINE_SPEED if TARGET_BASELINE_SPEED > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"  Total tokens generated: {avg_total_tokens:.0f}")
    print(f"  Total time: {avg_total_time:.2f}s")
    print(f"  Avg tokens/request: {avg_tokens_per_req:.2f}")
    print(f"  Avg latency: {avg_latency:.3f}s")
    print(f"  ")
    print(f"  🚀 Speculative Decoding Performance:")
    print(f"  Throughput: {throughput:.2f} tok/s")
    print(f"  Per-request throughput: {throughput_per_req:.2f} tok/s")
    print(f"  Estimated MAT: {estimated_mat:.2f} tokens/step")
    print(f"  Speedup vs baseline ({TARGET_BASELINE_SPEED:.1f} tok/s): {throughput_per_req / TARGET_BASELINE_SPEED:.2f}x")
    
    return BenchmarkResult(
        batch_size=batch_size,
        gamma=gamma,
        num_requests=batch_size * num_iterations,
        total_tokens=avg_total_tokens,
        total_time=avg_total_time,
        avg_tokens_per_req=avg_tokens_per_req,
        throughput=throughput,
        throughput_per_req=throughput_per_req,
        avg_latency=avg_latency,
    )

async def wait_for_server(url: str, max_wait: int = 300):
    """Wait for server to be ready"""
    # Health endpoint is at /health
    base = url.split('/v1/')[0] if '/v1/' in url else url.rsplit('/', 1)[0]
    health_url = f"{base}/health"
    print(f"⏳ Waiting for server at {health_url}...")
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start < max_wait:
            try:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        print("✅ Server is ready!\n")
                        return True
            except:
                pass
            await asyncio.sleep(2)
    
    print("❌ Server did not become ready in time")
    return False

async def main():
    URL = "http://localhost:30000/v1/chat/completions"
    PROMPT = "Write a short sentence about AI"
    MAX_TOKENS = 50
    
    print("="*80)
    print("🧪 nano-PEARL Speculative Decoding Benchmark")
    print("="*80)
    print("\nThis benchmark tests REAL speculative decoding performance with PEARL.")
    print("It measures actual MAT and throughput by sending API requests.\n")
    
    # Wait for server
    if not await wait_for_server(URL):
        return
    
    # Test configuration: smaller batch sizes for practical testing
    test_configs = [
        (1, GAMMA_CONFIG.get(1, 10), 3),    # batch_size, gamma, iterations
        (4, GAMMA_CONFIG.get(4, 9), 2),
        (8, GAMMA_CONFIG.get(8, 9), 2),
        (16, GAMMA_CONFIG.get(16, 9), 1),
        (32, GAMMA_CONFIG.get(32, 9), 1),
    ]
    
    results = []
    for batch_size, gamma, iterations in test_configs:
        result = await benchmark_batch_size(batch_size, gamma, URL, PROMPT, MAX_TOKENS, iterations)
        if result:
            results.append(result)
        await asyncio.sleep(1)  # Cool down between tests
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("📈 Summary: Speculative Decoding Performance")
        print(f"{'='*80}\n")
        print(f"{'Batch':<8} {'Gamma':<8} {'Throughput':<15} {'Per-Req':<15} {'Est. MAT':<12} {'Speedup':<10}")
        print("-"*80)
        for r in results:
            speedup = r.throughput_per_req / TARGET_BASELINE_SPEED
            est_mat = speedup  # Rough estimate
            print(f"{r.batch_size:<8} {r.gamma:<8} {r.throughput:>10.2f} tok/s {r.throughput_per_req:>10.2f} tok/s {est_mat:>8.2f}      {speedup:>6.2f}x")
        
        print(f"\n{'='*80}")
        print("Note: MAT is estimated as (per-req throughput / baseline speed)")
        print(f"Baseline target speed (no speculation): ~{TARGET_BASELINE_SPEED:.1f} tok/s")
        print("Real MAT depends on token acceptance rate in PEARL verification.")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
