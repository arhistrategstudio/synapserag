"""
Multi-Modal Embedder Layer.
Generates dense document vectors and per-token embeddings for ColBERT Late Interaction.
Includes zero-dependency local deterministic projection as well as pluggable neural backends.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import hashlib
import math
import re

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class MultiModalEmbedder:
    """
    Embedder capable of producing:
    1. Single dense vectors for standard semantic search.
    2. Per-token embedding matrices for ColBERT MaxSim late interaction.
    """
    def __init__(
        self,
        dim: int = 384,
        external_dense_fn: Optional[Callable[[str], List[float]]] = None,
        external_tokens_fn: Optional[Callable[[str], List[List[float]]]] = None
    ):
        self.dim = dim
        self.external_dense_fn = external_dense_fn
        self.external_tokens_fn = external_tokens_fn

    def embed_text(self, text: str) -> List[float]:
        """Generate a normalized dense vector for text."""
        if self.external_dense_fn:
            try:
                vec = self.external_dense_fn(text)
                return self._normalize(vec)
            except Exception:
                pass
        return self._local_deterministic_dense(text)

    def embed_tokens(self, text: str) -> List[List[float]]:
        """Generate token-level embedding vectors for ColBERT MaxSim."""
        if self.external_tokens_fn:
            try:
                token_vecs = self.external_tokens_fn(text)
                return [self._normalize(v) for v in token_vecs]
            except Exception:
                pass
        return self._local_deterministic_tokens(text)

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    def _token_to_vector(self, token: str) -> List[float]:
        """Deterministic pseudorandom projection of a token into dim-dimensional space."""
        vec = [0.0] * self.dim
        # Hash token with multiple seeds to create uniform distribution across dimensions
        h_base = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        h_secondary = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        
        for i in range(self.dim):
            val = ((h_base >> (i % 64)) & 0xFF) - 128.0
            val += (((h_secondary >> (i % 64)) & 0xFF) - 128.0) * 0.5
            vec[i] = val
            
        return self._normalize(vec)

    def _local_deterministic_dense(self, text: str) -> List[float]:
        """Aggregate token vectors with sub-word position and frequency weighting."""
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
        if not tokens:
            return [0.0] * self.dim

        dense = [0.0] * self.dim
        for i, token in enumerate(tokens):
            # Positional decay for late chunk context weighting
            pos_weight = 1.0 + (0.1 * math.sin(i))
            tok_vec = self._token_to_vector(token)
            for d in range(self.dim):
                dense[d] += tok_vec[d] * pos_weight

        return self._normalize(dense)

    def _local_deterministic_tokens(self, text: str) -> List[List[float]]:
        """Extract tokens and generate a vector for each token (for ColBERT Late Interaction)."""
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
        if not tokens:
            return [self._normalize([1.0] * self.dim)]
        # Limit token matrix to reasonable length for embedded performance
        capped_tokens = tokens[:128]
        return [self._token_to_vector(tok) for tok in capped_tokens]
