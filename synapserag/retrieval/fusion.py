"""
Reciprocal Rank Fusion (RRF) for Multi-Channel Retrieval.
Fuses ranking outputs from dense, graph-associative, and sparse channels.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict


class ReciprocalRankFusion:
    """
    Weighted Reciprocal Rank Fusion:
    RRF(d) = sum_{c in channels} w_c * 1 / (k + rank_c(d))
    """
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        dense_results: List[Tuple[str, float]],
        graph_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        weight_dense: float = 0.40,
        weight_graph: float = 0.35,
        weight_sparse: float = 0.25,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Merges result lists into unified ranked output.
        Returns: List of (chunk_id, fused_score, {channel: individual_score})
        """
        combined_scores: Dict[str, float] = defaultdict(float)
        channel_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: {"dense": 0.0, "graph": 0.0, "sparse": 0.0})

        # Process Dense
        for rank, (cid, score) in enumerate(dense_results, start=1):
            rrf_val = weight_dense * (1.0 / (self.k + rank))
            combined_scores[cid] += rrf_val
            channel_scores[cid]["dense"] = score

        # Process Graph
        for rank, (cid, score) in enumerate(graph_results, start=1):
            rrf_val = weight_graph * (1.0 / (self.k + rank))
            combined_scores[cid] += rrf_val
            channel_scores[cid]["graph"] = score

        # Process Sparse
        for rank, (cid, score) in enumerate(sparse_results, start=1):
            rrf_val = weight_sparse * (1.0 / (self.k + rank))
            combined_scores[cid] += rrf_val
            channel_scores[cid]["sparse"] = score

        # Normalize into a [0, 1] confidence scale: a document ranked #1 across every
        # weighted channel scores 1.0. Without this, raw RRF scores top out around
        # 1/(k+1) (~0.016 for k=60), which is meaningless compared against an
        # absolute confidence threshold downstream (e.g. the circuit breaker).
        max_possible = (weight_dense + weight_graph + weight_sparse) / (self.k + 1)
        if max_possible > 0:
            combined_scores = {cid: score / max_possible for cid, score in combined_scores.items()}

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(cid, score, channel_scores[cid]) for cid, score in ranked[:top_k]]
