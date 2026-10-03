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
    # Absolute floor on the raw semantic similarity of the top hybrid match. Guards against
    # corpora where RRF rank-consensus alone looks perfect (e.g. a single unrelated chunk
    # trivially ranks #1 in every channel) although the match has no real relevance.
    #
    # None (default) = pick a floor that matches the active embedding setup:
    #   * neural backend, dense cosine (enable_late_chunking=False): 0.40
    #   * neural backend, ColBERT MaxSim (enable_late_chunking=True): 0.60
    #   * zero-dependency hash embedder:                               0.22
    # The neural values were calibrated for paraphrase-multilingual-MiniLM-L12-v2 on a small
    # Serbian legal sample (10 in-domain / 5 out-of-domain queries, see CHANGELOG 0.2.0):
    # in-domain top matches scored >= 0.52 (dense) / >= 0.71 (MaxSim), unrelated queries
    # mostly 0.18-0.36 (dense) / 0.24-0.55 (MaxSim). This is a provisional calibration, not a
    # guarantee: a query that shares vocabulary with the corpus can still pass the gate.
    # Set an explicit float to override.
    min_semantic_similarity: Optional[float] = None
    enable_circuit_breaker: bool = True
    
    # Embedding Dimension (Default 384 for MiniLM/ColBERT, configurable)
    embedding_dim: int = 384

    # Embedding backend: "auto" tries sentence-transformers and falls back to the
    # zero-dependency deterministic hash embedder if the library/model is unavailable.
    # "hash" forces the deterministic fallback. "sentence-transformers" forces the
    # neural backend and raises if it cannot be loaded.
    embedding_backend: str = "auto"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def resolve_semantic_floor(self, neural_backend: bool, late_interaction: bool) -> float:
        """Effective absolute similarity floor for the circuit breaker (see min_semantic_similarity)."""
        if self.min_semantic_similarity is not None:
            return self.min_semantic_similarity
        if not neural_backend:
            return 0.22
        return 0.60 if late_interaction else 0.40

    def get_storage_path(self) -> Path:
        p = Path(self.storage_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
