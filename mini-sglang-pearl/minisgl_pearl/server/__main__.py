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
from typing import AsyncGenerator
import json
import time
import uuid

from minisgl_pearl.engine.pearl_engine import PEARLEngine
# minisgl is now installed via pip, so we can import directly
from minisgl.core import SamplingParams


# Global engine  instance
engine: PEARLEngine = None

# Create FastAPI app
app = FastAPI(title="Mini-SGLang-PEARL Server")


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def create_completion(request: Request):
    """OpenAI-compatible completions endpoint"""
    request_dict = await request.json()
    
    # Parse request
    prompt = request_dict.get("prompt")
    if not prompt and "messages" in request_dict:
        # Convert messages to prompt (simple concatenation for now)
        messages = request_dict["messages"]
        if isinstance(messages, list):
            prompt = "\n".join([str(m.get("content", "")) for m in messages if isinstance(m, dict)])
        elif isinstance(messages, str):
            prompt = messages
            
    if not prompt:
        prompt = ""

    max_tokens = request_dict.get("max_tokens", 256)
    temperature = request_dict.get("temperature", 1.0)
    top_k = request_dict.get("top_k", 1)
    stream = request_dict.get("stream", False)
    
    # Create sampling params (mini-sglang format)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        ignore_eos=False,
    )
    
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
    
    if stream:
        # Streaming response
        async def generate_stream() -> AsyncGenerator[str, None]:
            while engine.get_num_unfinished_requests() > 0:
                outputs = await engine.generate()
                for output in outputs:
                    if output['request_id'] == request_id:
                        chunk = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": "pearl",
                            "choices": [{
                                "text": output['text'],
                                "index": 0,
                                "finish_reason": "stop" if output['finished'] else None,
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
                        if output['finished']:
                            yield "data: [DONE]\n\n"
                            return
                
                await asyncio.sleep(0.01)
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    else:
        # Non-streaming response
        while engine.get_num_unfinished_requests() > 0:
            outputs = await engine.generate()
            for output in outputs:
                if output['request_id'] == request_id:
                    return JSONResponse({
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": "pearl",
                        "choices": [{
                            "text": output['text'],
                            "index": 0,
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": len(prompt_token_ids),
                            "completion_tokens": output['num_tokens'],
                            "total_tokens": len(prompt_token_ids) + output['num_tokens'],
                        }
                    })


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
    print("=" * 60)
    
    # Initialize PEARL engine with nano-PEARL KV cache
    engine = PEARLEngine(
        model_path=args.model,
        draft_model_path=args.draft_model,
        draft_tp_size=draft_tp,
        target_tp_size=target_tp,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
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
