"""
Neuro-Associative Graph Retriever using Vector-Biased Personalized PageRank.
Propagates semantic relevance through multi-hop entity pathways in a single step.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from ..storage.graph_store import EmbeddedGraphStore


class NeuroAssociativeRetriever:
    """
    Executes graph-level associative retrieval.
    Discovers indirectly connected chunks and concepts (multi-hop reasoning)
    without relying on multi-turn LLM chains.
    """
    def __init__(self, graph_store: EmbeddedGraphStore, damping: float = 0.85, vector_bias_weight: float = 0.5):
        self.graph_store = graph_store
        self.damping = damping
        self.vector_bias_weight = vector_bias_weight

    def retrieve(
        self,
        query_keywords: List[str],
        query_vector: Optional[List[float]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        1. Identifies seed nodes from query keywords.
        2. Runs vector-biased Personalized PageRank over the knowledge graph.
        3. Projects node activation back to chunks.
        """
        # Step 1: Find seed nodes
        seed_node_ids = self.graph_store.find_nodes_by_name(query_keywords)
        if not seed_node_ids:
            return []

        # Step 2: Compute Vector-Biased PPR
        node_scores = self.graph_store.compute_vector_biased_ppr(
            seed_nodes=seed_node_ids,
            query_vector=query_vector,
            damping=self.damping,
            vector_bias_weight=self.vector_bias_weight
        )

        # Step 3: Map node activation back to text chunks
        chunk_scores = self.graph_store.get_chunk_scores_from_ppr(node_scores)
        return chunk_scores[:top_k]
