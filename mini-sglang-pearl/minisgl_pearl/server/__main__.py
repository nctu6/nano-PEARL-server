#!/usr/bin/env python3
"""
Launch mini-sglang-pearl server

Combines:
- mini-sglang's continuous batching and API server
- nano-PEARL's KV cache and speculative decoding engine

Usage:
    python -m minisgl_pearl.server \
        --model /path/to/target/model \
        --draft-model /path/to/draft/model \
        --tp 2 \
        --port 30000
"""

import argparse
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator, Dict, Optional
import json
import time
import uuid

from minisgl_pearl.engine.pearl_engine import PEARLEngine
# minisgl is now installed via pip, so we can import directly
from minisgl.core import SamplingParams


# Global engine  instance
engine: PEARLEngine = None
request_queues: Dict[str, asyncio.Queue] = {}
dispatcher_task: Optional[asyncio.Task] = None

# Create FastAPI app
app = FastAPI(title="Mini-SGLang-PEARL Server")


async def stream_dispatcher():
    """Single fan-out loop so stream/non-stream requests share one generator."""
    global dispatcher_task
    try:
        while True:
            # Exit when no pending work and no listeners
            unfinished = engine.get_num_unfinished_requests()
            if not request_queues and unfinished == 0:
                break
            # If engine已無未完成但仍有等待的queue，強制送終止哨兵避免卡死
            if unfinished == 0 and request_queues:
                for q in list(request_queues.values()):
                    try:
                        q.put_nowait(None)
                    except Exception:
                        pass
                request_queues.clear()
                break

            outputs = await engine.generate()
            if not outputs:
                await asyncio.sleep(0.001)
                continue

            for output in outputs:
                queue = request_queues.get(output["request_id"])
                if queue:
                    await queue.put(output)
                    if output.get("finished"):
                        await queue.put(None)
    except Exception as e:
        print(f"❌ Stream dispatcher error: {e}")
    finally:
        # Signal all listeners to exit if dispatcher stops unexpectedly
        for q in list(request_queues.values()):
            try:
                q.put_nowait(None)
            except Exception:
                pass
        dispatcher_task = None


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def create_completion(request: Request):
    """OpenAI-compatible completions endpoint"""
    request_dict = await request.json()
    
    # Get model name (optional for now, but parsed for compatibility)
    model = request_dict.get("model", "pearl")
    
    # Parse prompt from either 'prompt' or 'messages' format
    prompt = request_dict.get("prompt")
    if not prompt and "messages" in request_dict:
        # Convert messages array to prompt (OpenAI chat format)
        messages = request_dict["messages"]
        if isinstance(messages, list):
            # Use tokenizer's chat template if available
            if hasattr(engine.engine, "tokenizer") and hasattr(engine.engine.tokenizer, "apply_chat_template"):
                try:
                    prompt = engine.engine.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"Error applying chat template: {e}")
                    # Fallback to naive formatting if template fails
                    prompt_parts = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            if role == "system":
                                prompt_parts.append(f"System: {content}")
                            elif role == "user":
                                prompt_parts.append(f"User: {content}")
                            elif role == "assistant":
                                prompt_parts.append(f"Assistant: {content}")
                            else:
                                prompt_parts.append(content)
                    prompt = "\n".join(prompt_parts)
            else:
                # Fallback to naive formatting
                prompt_parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "system":
                            prompt_parts.append(f"System: {content}")
                        elif role == "user":
                            prompt_parts.append(f"User: {content}")
                        elif role == "assistant":
                            prompt_parts.append(f"Assistant: {content}")
                        else:
                            prompt_parts.append(content)
                prompt = "\n".join(prompt_parts)

        elif isinstance(messages, str):
            prompt = messages
            
    if not prompt:
        return JSONResponse(
            {"error": "Either 'prompt' or 'messages' must be provided"},
            status_code=400
        )

    # Parse parameters with OpenAI defaults
    max_tokens = request_dict.get("max_tokens", 256)
    temperature = request_dict.get("temperature", 1.0)
    top_k = request_dict.get("top_k", -1)
    top_p = request_dict.get("top_p", 1.0)
    frequency_penalty = request_dict.get("frequency_penalty", 0.0)
    presence_penalty = request_dict.get("presence_penalty", 0.0)
    stream = request_dict.get("stream", False)
    
    # Create sampling params (mini-sglang format - doesn't support top_p natively)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        ignore_eos=False,
    )
    # Manually add top_p and penalties as attributes
    sampling_params.top_p = top_p
    sampling_params.frequency_penalty = frequency_penalty
    sampling_params.presence_penalty = presence_penalty
    
    # Generate request ID
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    
    # Tokenize prompt (simple split for now, should use tokenizer)
    prompt_token_ids = list(range(len(prompt.split())))  # Placeholder
    
    # Add request to engine
    engine.add_request(
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    # Register queue for this request and ensure dispatcher is running
    queue: asyncio.Queue = asyncio.Queue()
    request_queues[request_id] = queue
    global dispatcher_task
    if dispatcher_task is None or dispatcher_task.done():
        dispatcher_task = asyncio.create_task(stream_dispatcher())
    
    if stream:
        # Streaming response
        async def generate_stream() -> AsyncGenerator[str, None]:
            try:
                while True:
                    output = await queue.get()
                    if output is None:
                        break
                    if output["request_id"] != request_id:
                        continue

                    # Use text_diff if available for proper streaming
                    chunk_text = output.get('text_diff', output.get('text', ''))
                    is_finished = output.get('finished')
                    
                    if chunk_text:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": chunk_text,
                                    "reasoning_content": None
                                },
                                "logprobs": None,
                                "finish_reason": "stop" if is_finished else None,
                                "stop_reason": None if is_finished else None,
                                "token_ids": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    if is_finished:
                        # Send usage chunk
                        usage_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": len(prompt_token_ids),
                                "total_tokens": len(prompt_token_ids) + output['num_tokens'],
                                "completion_tokens": output['num_tokens']
                            }
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
            finally:
                request_queues.pop(request_id, None)
         
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
     
    else:
        # Non-streaming response
        final_output = None
        try:
            while True:
                output = await queue.get()
                if output is None:
                    break
                if output["request_id"] != request_id:
                    continue
                final_output = output
                if output.get('finished'):
                    break
        finally:
            request_queues.pop(request_id, None)

        if final_output:
            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_output['text'],
                        "refusal": None,
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": [],
                        "reasoning": None,
                        "reasoning_content": None
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                    "stop_reason": None,
                    "token_ids": None
                }],
                "service_tier": None,
                "system_fingerprint": None,
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "total_tokens": len(prompt_token_ids) + final_output['num_tokens'],
                    "completion_tokens": final_output['num_tokens'],
                    "prompt_tokens_details": None
                },
                "prompt_logprobs": None,
                "prompt_token_ids": None,
                "kv_transfer_params": None
            })
        else:
            # Should not happen if engine works correctly
            return JSONResponse({"error": "Request failed"}, status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "engine": "PEARL"}


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": "pearl",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "minisgl-pearl",
        }]
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Mini-SGLang-PEARL Server")
    parser.add_argument("--model", type=str, required=True, help="Path to target model")
    parser.add_argument("--draft-model", type=str, required=True, help="Path to draft model")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (for backward compatibility, sets target-tp)")
    parser.add_argument("--draft-tp", type=int, default=None, help="Draft model tensor parallel size (default: 1)")
    parser.add_argument("--target-tp", type=int, default=None, help="Target model tensor parallel size (default: same as --tp)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Max number of sequences")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384, help="Max batched tokens")
    parser.add_argument("--benchmark", action="store_true", help="Enable speculative decoding benchmark after Auto Set Gamma")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization ratio (default: 0.9)")
    
    return parser.parse_args()


def main():
    global engine
    
    args = parse_args()
    
    # Set default TP sizes
    draft_tp = args.draft_tp if args.draft_tp is not None else 1
    target_tp = args.target_tp if args.target_tp is not None else args.tp
    
    print("=" * 60)
    print("🚀 Starting Mini-SGLang-PEARL Server")
    print("=" * 60)
    print(f"Target Model: {args.model}")
    print(f"Draft Model: {args.draft_model}")
    print(f"Draft TP: {draft_tp}")
    print(f"Target TP: {target_tp}")
    print(f"Total GPUs needed: {draft_tp + target_tp}")
    print(f"Server: {args.host}:{args.port}")
    print(f"GPU Mem Util: {args.gpu_memory_utilization}")
    print("=" * 60)
    
    # Initialize PEARL engine with nano-PEARL KV cache
    engine = PEARLEngine(
        model_path=args.model,
        draft_model_path=args.draft_model,
        draft_tp_size=draft_tp,
        target_tp_size=target_tp,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_benchmark=args.benchmark,
    )
    
    print("\n✅ PEARL Engine initialized with nano-PEARL KV cache")
    print(f"🌐 Server running at http://{args.host}:{args.port}")
    print("📚 API: /v1/completions (OpenAI-compatible)")
    print("=" * 60)
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
