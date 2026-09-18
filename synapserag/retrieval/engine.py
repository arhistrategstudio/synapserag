"""
Tri-Brain Unified Retrieval Engine.
Harmonizes Dense/ColBERT, Neuro-Associative Graph, and BM25 Sparse matching.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from ..types import Chunk, RetrievalMatch, RetrievalMode
from ..config import SynapseConfig
from ..storage.vector_store import EmbeddedVectorStore
from ..storage.graph_store import EmbeddedGraphStore
from ..storage.sparse_store import EmbeddedSparseStore
from .ppr import NeuroAssociativeRetriever
from .fusion import ReciprocalRankFusion


class TriBrainRetriever:
    """
    Coordinates and executes searches across all three cognitive brains.
    """
    def __init__(
        self,
        vector_store: EmbeddedVectorStore,
        graph_store: EmbeddedGraphStore,
        sparse_store: EmbeddedSparseStore,
        config: SynapseConfig
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.sparse_store = sparse_store
        self.config = config
        
        self.graph_retriever = NeuroAssociativeRetriever(
            graph_store=self.graph_store,
            damping=self.config.ppr_damping_factor,
            vector_bias_weight=self.config.vector_bias_weight
        )
        self.fusion = ReciprocalRankFusion(k=self.config.rrf_k)

    def retrieve(
        self,
        query: str,
        query_vector: Optional[List[float]],
        query_token_embeddings: Optional[List[List[float]]],
        keywords: List[str],
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 10
    ) -> List[RetrievalMatch]:
        """
        Executes retrieval according to specified mode, combining cognitive stores.
        """
        if mode == RetrievalMode.LATE_INTERACTION and query_token_embeddings:
            raw_matches = self.vector_store.search_late_interaction(query_token_embeddings, top_k=top_k)
            return self._build_matches(raw_matches, channel="late_interaction")

        if mode == RetrievalMode.GRAPH_HOP:
            raw_matches = self.graph_retriever.retrieve(
                query_keywords=keywords,
                query_vector=query_vector,
                top_k=top_k
            )
            return self._build_matches(raw_matches, channel="graph_hop")

        if mode == RetrievalMode.LEXICAL:
            raw_matches = self.sparse_store.search(query, top_k=top_k)
            return self._build_matches(raw_matches, channel="lexical")

        # HYBRID / AUTO Mode: Tri-Brain Execution & Fusion
        # 1. Brain A (Dense / Late Interaction)
        if query_token_embeddings and len(self.vector_store.token_embeddings) > 0:
            dense_results = self.vector_store.search_late_interaction(query_token_embeddings, top_k=top_k * 2)
        elif query_vector:
            dense_results = self.vector_store.search_dense(query_vector, top_k=top_k * 2)
        else:
            dense_results = []

        # 2. Brain B (Neuro-Associative Graph PPR)
        graph_results = self.graph_retriever.retrieve(
            query_keywords=keywords,
            query_vector=query_vector,
            top_k=top_k * 2
        )

        # 3. Brain C (BM25 Sparse)
        sparse_results = self.sparse_store.search(query, top_k=top_k * 2)

        # 4. Fuse using RRF
        fused = self.fusion.fuse(
            dense_results=dense_results,
            graph_results=graph_results,
            sparse_results=sparse_results,
            weight_dense=self.config.weight_dense,
            weight_graph=self.config.weight_graph,
            weight_sparse=self.config.weight_sparse,
            top_k=top_k
        )

        matches: List[RetrievalMatch] = []
        for cid, score, ch_scores in fused:
            chunk = self.vector_store.get_chunk(cid)
            if not chunk:
                continue
            
            explanation = (
                f"Tri-Brain Fusion: Dense={ch_scores['dense']:.3f}, "
                f"Graph={ch_scores['graph']:.3f}, Sparse={ch_scores['sparse']:.3f}"
            )
            matches.append(
                RetrievalMatch(
                    chunk=chunk,
                    score=score,
                    dense_score=ch_scores["dense"],
                    graph_score=ch_scores["graph"],
                    sparse_score=ch_scores["sparse"],
                    channel="hybrid",
                    explanation=explanation
                )
            )

        return matches

    def _build_matches(self, raw_matches: List[Tuple[str, float]], channel: str) -> List[RetrievalMatch]:
        matches = []
        for cid, score in raw_matches:
            chunk = self.vector_store.get_chunk(cid)
            if chunk:
                matches.append(RetrievalMatch(chunk=chunk, score=score, channel=channel))
        return matches
