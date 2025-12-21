"""
Mini-SGLang + nano-PEARL Integration

Architecture:
- mini-sglang: API server + continuous batching scheduler
- nano-PEARL: KV cache + speculative decoding engine

Key Point: Uses nano-PEARL's native KV cache, NOT mini-sglang's page cache.
"""

from .engine.pearl_engine import PEARLEngine

__version__ = "0.1.0"
__all__ = ["PEARLEngine"]
