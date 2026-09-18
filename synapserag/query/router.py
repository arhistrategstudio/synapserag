"""
Adaptive Query Router.
Analyzes syntax and intent to dispatch queries to the optimal retrieval channel.
"""

from __future__ import annotations
import re

from ..types import RetrievalMode


class AdaptiveQueryRouter:
    """
    Classifies queries to dynamically assign the most effective retrieval strategy.
    """
    @staticmethod
    def route(query: str) -> RetrievalMode:
        q = query.strip()
        
        # 1. Exact identifier / Code syntax / Quoted match -> LEXICAL (BM25)
        if re.search(r"`[^`]+`|\"[^\"]+\"", q) or re.search(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-z]+[A-Z][a-zA-Z0-9]*\b", q):
            return RetrievalMode.LEXICAL

        # 2. Multi-hop questions ("how does X relate to Y", "impact of X on Y", "where does X call Y") -> GRAPH_HOP
        multi_hop_patterns = [
            r"\b(how|why)\s+does\s+.+\s+(call|trigger|affect|relate|depend|connect)",
            r"\brelationship\s+between\b",
            r"\bdependency\s+chain\b",
            r"\bcompare\b.+\bwith\b"
        ]
        for pattern in multi_hop_patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return RetrievalMode.GRAPH_HOP

        # 3. Micro-level exact passage / precision lookup -> LATE_INTERACTION
        if len(q.split()) > 15:
            return RetrievalMode.LATE_INTERACTION

        # 4. Default to Tri-Brain HYBRID for comprehensive multi-channel coverage
        return RetrievalMode.HYBRID
