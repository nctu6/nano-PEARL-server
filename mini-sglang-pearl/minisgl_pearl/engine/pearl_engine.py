"""
PEARL Engine Wrapper for mini-sglang Integration

Integrates nano-PEARL's speculative decoding engine with mini-sglang's
continuous batching scheduler and API infrastructure.

Key Architecture:
- mini-sglang: Request scheduling, API layer, continuous batching
- nano-PEARL: Model execution, KV cache management, speculative decoding
"""

import asyncio
from typing import List, Optional, Dict, Any
import torch
from dataclasses import dataclass

# Import from mini-sglang for API compatibility
import sys
sys.path.insert(0, '/app/mini-sglang/python')
from minisgl.core import SamplingParams as MiniSGLSamplingParams

# Import from nano-PEARL for engine
sys.path.insert(0, '/app/nano-PEARL')
from nano_pearl import PEARLConfig, PEARLEngine as NanoPEARLEngine
from nano_pearl import SamplingParams as PEARLSamplingParams


@dataclass
class BatchRequest:
    """Request format compatible with mini-sglang's scheduler"""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: MiniSGLSamplingParams
    arrival_time: float


class PEARLEngine:
    """
    PEARL Engine that works with mini-sglang's continuous batching.
    
    - Uses nano-PEARL's KV cache (NOT mini-sglang's page cache)
    - Accepts mini-sglang scheduler's batch format
    - Provides continuous batching without BATCH_MAX_WAIT delays
    """
    
    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        draft_tp_size: int = 1,
        target_tp_size: int = 1,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        Initialize PEARL engine with nano-PEARL's native KV cache.
        
        Args:
            model_path: Path to target model
            draft_model_path: Path to draft model for speculative decoding
            draft_tp_size: Tensor parallel size for draft model
            target_tp_size: Tensor parallel size for target model
            max_num_batched_tokens: Max tokens in a batch (nano-PEARL config)
            max_num_seqs: Max sequences in a batch (nano-PEARL config)
            max_model_len: Max sequence length (nano-PEARL config)
            gpu_memory_utilization: GPU memory usage ratio
        """
        
        print("🚀 Initializing PEARL Engine with nano-PEARL KV cache...")
        
        # Create nano-PEARL configuration
        self.pearl_config = PEARLConfig(
            draft_model_path=draft_model_path,
            target_model_path=model_path,
            draft_tensor_parallel_size=draft_tp_size,
            target_tensor_parallel_size=target_tp_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        # Initialize nano-PEARL engine
        # This includes nano-PEARL's native KV cache management
        self.engine = NanoPEARLEngine(self.pearl_config)
        
        # Request tracking for continuous batching
        self.pending_requests: Dict[str, BatchRequest] = {}
        self.running_requests: Dict[str, BatchRequest] = {}
        
        print("✅ PEARL Engine initialized with nano-PEARL KV cache")
    
    def add_request(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: MiniSGLSamplingParams,
    ) -> None:
        """
        Add a request to the engine (mini-sglang compatible interface).
        Requests are processed immediately via continuous batching.
        """
        import time
        
        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time=time.time(),
        )
        
        self.pending_requests[request_id] = request
        print(f"📝 Added request {request_id}, queue size: {len(self.pending_requests)}")
    
    def _convert_sampling_params(
        self, 
        minisgl_params: MiniSGLSamplingParams
    ) -> PEARLSamplingParams:
        """Convert mini-sglang sampling params to nano-PEARL format"""
        return PEARLSamplingParams(
            temperature=minisgl_params.temperature,
            top_k=minisgl_params.top_k,
            max_tokens=minisgl_params.max_tokens,
            ignore_eos=minisgl_params.ignore_eos,
        )
    
    async def generate(self) -> List[Dict[str, Any]]:
        """
        Generate tokens using PEARL speculative decoding.
        
        Uses continuous batching - processes requests immediately
        without waiting for batch to fill.
        
        Returns:
            List of generation outputs with request_id and tokens
        """
        if not self.pending_requests and not self.running_requests:
            await asyncio.sleep(0.001)  # Minimal wait, not 1 second!
            return []
        
        # Move pending to running (continuous batching)
        while self.pending_requests:
            req_id, req = self.pending_requests.popitem()
            self.running_requests[req_id] = req
        
        if not self.running_requests:
            return []
        
        # Prepare batch for nano-PEARL
        prompts = []
        sampling_params_list = []
        request_ids = []
        
        for req_id, req in self.running_requests.items():
            prompts.append(req.prompt)
            pearl_params = self._convert_sampling_params(req.sampling_params)
            sampling_params_list.append(pearl_params)
            request_ids.append(req_id)
        
        # Clear requests for nano-PEARL
        self.engine.scheduler.clear()
        
        # Add all requests to nano-PEARL engine
        for prompt, params in zip(prompts, sampling_params_list):
            self.engine.add_request(prompt, params)
        
        # Run PEARL generation with nano-PEARL's KV cache
        print(f"⚡ Running PEARL generation for {len(prompts)} requests...")
        text_outputs, token_counts, accept_lengths, mean_acceptance_tokens = \
            await asyncio.to_thread(self.engine.generate)
        
        # Format outputs
        outputs = []
        for i, (req_id, text, tokens) in enumerate(
            zip(request_ids, text_outputs, token_counts)
        ):
            outputs.append({
                'request_id': req_id,
                'text': text,
                'num_tokens': tokens,
                'finished': True,  # Full generation for now
            })
            
            # Remove completed request
            if req_id in self.running_requests:
                del self.running_requests[req_id]
        
        print(f"✅ Generated {len(outputs)} outputs, MAT: {mean_acceptance_tokens:.2f}")
        return outputs
    
    def abort_request(self, request_id: str) -> None:
        """Abort a request"""
        self.pending_requests.pop(request_id, None)
        self.running_requests.pop(request_id, None)
    
    def get_num_unfinished_requests(self) -> int:
        """Get number of pending/running requests"""
        return len(self.pending_requests) + len(self.running_requests)
