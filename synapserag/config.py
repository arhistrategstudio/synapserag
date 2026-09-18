"""
SynapseRAG Configuration Schema.
Supports zero-configuration local embedded defaults as well as production customization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SynapseConfig:
    """Master configuration for the SynapseRAG engine."""
    # Storage settings
    storage_dir: str = "./data/synapse_storage"
    persist_on_write: bool = True
    
    # Ingestion & Chunking parameters
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 64
    enable_late_chunking: bool = True
    enable_contextual_retrieval: bool = True
    
    # Tri-Brain Weights for Reciprocal Rank Fusion (RRF)
    weight_dense: float = 0.40
    weight_graph: float = 0.35
    weight_sparse: float = 0.25
    rrf_k: int = 60 # Standard RRF smoothing constant
    
    # Neuro-Associative Graph (Personalized PageRank) parameters
    ppr_damping_factor: float = 0.85
    ppr_max_iterations: int = 30
    ppr_tolerance: float = 1e-6
    vector_bias_weight: float = 0.60 # Influence of semantic vector on graph transition probabilities
    
    # Query Intelligence & Clue Engine (MemoRAG System 1)
    enable_clue_generation: bool = True
    max_speculative_clues: int = 3
    
    # Verification & Circuit Breaker
    min_confidence_threshold: float = 0.25
    # Absolute floor on raw dense cosine similarity for hybrid-mode top matches.
    # Guards against small corpora where RRF rank-consensus alone can look perfect
    # (e.g. a single unrelated chunk trivially ranks #1 in every channel) even
    # though the match has no real semantic relevance to the query.
    # Calibrated against sentence-transformers/all-MiniLM-L6-v2, whose "noise
    # floor" for genuinely unrelated sentence pairs sits around 0.15-0.2 while
    # real matches typically score 0.5+. The zero-dependency hash embedder has a
    # noisier, higher floor (~0.35-0.4 for short unrelated texts), so this gate
    # is weaker (but still harmless — real matches score even higher, ~0.8) when
    # running without the neural backend.
    min_semantic_similarity: float = 0.22
    enable_circuit_breaker: bool = True
    
    # Embedding Dimension (Default 384 for MiniLM/ColBERT, configurable)
    embedding_dim: int = 384

    # Embedding backend: "auto" tries sentence-transformers and falls back to the
    # zero-dependency deterministic hash embedder if the library/model is unavailable.
    # "hash" forces the deterministic fallback. "sentence-transformers" forces the
    # neural backend and raises if it cannot be loaded.
    embedding_backend: str = "auto"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def get_storage_path(self) -> Path:
        p = Path(self.storage_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
