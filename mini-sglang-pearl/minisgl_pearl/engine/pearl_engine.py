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
import os
import time
import asyncio
from typing import List, Dict, Any, Union
import uuid

# ... imports ...
# Get the base directory of nano-PEARL-server
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add mini-sglang to path (if not already installed)
# minisgl_path = os.path.join(BASE_DIR, 'mini-sglang', 'python')
# if os.path.exists(minisgl_path):
#     sys.path.insert(0, minisgl_path)

# minisgl is now installed via pip, so we can import directly
from minisgl.core import SamplingParams as MiniSGLSamplingParams

# Import from nano-PEARL for engine
nano_pearl_path = os.path.join(BASE_DIR, 'nano-PEARL')
if os.path.exists(nano_pearl_path):
    sys.path.insert(0, nano_pearl_path)
from nano_pearl import PEARLConfig, PEARLEngine as NanoPEARLEngine
from nano_pearl import SamplingParams as PEARLSamplingParams
from nano_pearl.utils.pearl_logger import logger


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
        
        # Streaming state
        self.generator = None
        self.previous_texts = {}
        self.processed_acc_indices = {}
        self.generator_start_time = 0
        self.total_gen_tokens = 0
        self._lock = None
        
        print("✅ PEARL Engine initialized with nano-PEARL KV cache")

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
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
            # top_k is not supported in nano-PEARL yet
            max_tokens=minisgl_params.max_tokens,
            ignore_eos=minisgl_params.ignore_eos,
        )
    
    async def generate(self) -> List[Dict[str, Any]]:
        """
        Stream generation output using step-method.
        Returns partial outputs or empty list if no new data.
        """
        async with self.lock:
            # 1. Start new stream if idle and have pending
            if self.generator is None and self.pending_requests:
                # Move all pending to running
                while self.pending_requests:
                    # Dict popitem is LIFO (Stack). For FIFO use iterator
                    # To be safe and compatible with original logic, we accept popitem
                    # But ideally we want FIFO. Let's use list(keys) to be proper.
                    keys = list(self.pending_requests.keys())
                    for k in keys:
                        self.running_requests[k] = self.pending_requests.pop(k)
                
                # Prepare engine args
                prompts = []
                sampling_params_list = []
                request_ids = []
                
                for req_id, req in self.running_requests.items():
                    prompts.append(req.prompt)
                    pearl_params = self._convert_sampling_params(req.sampling_params)
                    sampling_params_list.append(pearl_params)
                    request_ids.append(req_id)
                    
                    # Reset state
                    self.previous_texts[req_id] = ""
                    self.processed_acc_indices[req_id] = 0
                
                # Add to engine
                for prompt, params, req_id in zip(prompts, sampling_params_list, request_ids):
                    self.engine.add_request(prompt, params, request_id=req_id)
                
                self.total_steps = 0  # <--- Initialize step counter
                print(f"⚡ Starting generation stream for {len(prompts)} requests...")
                self.generator = self.engine.stream_generate() # Returns iterator
                self.generator_start_time = time.time()
                self.total_gen_tokens = 0
            
            if self.generator is None:
                await asyncio.sleep(0.001)
                return []

            # 2. Step
            try:
                def _next():
                    return next(self.generator)
                
                # Run one step in thread
                output_tuple, batch_finished = await asyncio.to_thread(_next)
                self.total_steps += 1  # <--- Increment step counter
                
                # Unpack: seq_id, text, tokens, acc_counts, finished_flags
                seq_ids, text_outputs, token_counts, acc_counts, finished_flags = output_tuple
                
                results = []
                
                for i, seq_id in enumerate(seq_ids):
                    full_text = text_outputs[i]
                    is_fin = finished_flags[i]
                    
                    # Diff logic
                    prev_text = self.previous_texts.get(seq_id, "")
                    diff_text = full_text[len(prev_text):]
                    self.previous_texts[seq_id] = full_text
                    
                    # Throughput update
                    acc_list = acc_counts[i]
                    processed_idx = self.processed_acc_indices.get(seq_id, 0)
                    new_acc_list = acc_list[processed_idx:]
                    new_valid_tokens = sum(new_acc_list)
                    self.total_gen_tokens += new_valid_tokens
                    self.processed_acc_indices[seq_id] = len(acc_list)
                    
                    results.append({
                        'request_id': seq_id,
                        'text': full_text,
                        'text_diff': diff_text, 
                        'num_tokens': token_counts[i],
                        'finished': is_fin
                    })
                    
                    if is_fin:
                        # Clean up this request
                        if seq_id in self.running_requests:
                            del self.running_requests[seq_id]
                        if seq_id in self.previous_texts:
                            del self.previous_texts[seq_id]
                        if seq_id in self.processed_acc_indices:
                            del self.processed_acc_indices[seq_id]

                if batch_finished:
                    elapsed = time.time() - self.generator_start_time
                    tps = self.total_gen_tokens / elapsed if elapsed > 0 else 0
                    
                    # Calculate MAT (Mean Accepted Tokens per step)
                    mat = self.total_gen_tokens / self.total_steps if self.total_steps > 0 else 0
                    
                    print(f"✅ Batch finished. Throughput: {tps:.2f} tok/s (Total: {self.total_gen_tokens}), MAT: {mat:.2f}")
                    self.generator = None
                
                return results

            except StopIteration:
                self.generator = None
                return []
            except Exception as e:
                print(f"❌ Error in generation step: {e}")
                import traceback
                traceback.print_exc()
                self.generator = None
                return []
    
    def abort_request(self, request_id: str) -> None:
        """Abort a request"""
        self.pending_requests.pop(request_id, None)
        self.running_requests.pop(request_id, None)
    
    def get_num_unfinished_requests(self) -> int:
        """Get number of pending/running requests"""
        return len(self.pending_requests) + len(self.running_requests)
