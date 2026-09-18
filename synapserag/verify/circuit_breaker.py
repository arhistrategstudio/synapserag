"""
Hallucination Circuit Breaker.
Monitors retrieval confidence and blocks baseless generation if source evidence is insufficient.
"""

from __future__ import annotations
from typing import List, Tuple

from ..types import RetrievalMatch


class HallucinationCircuitBreaker:
    """
    Evaluates grounding confidence of retrieval results.
    If evidence is missing or scores fall below the safety threshold,
    triggers a circuit-breaker to alert the agent to abstain or fallback to web search.
    """
    def __init__(self, min_confidence_threshold: float = 0.05, min_semantic_similarity: float = 0.22):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_semantic_similarity = min_semantic_similarity

    def evaluate(self, matches: List[RetrievalMatch]) -> Tuple[bool, str]:
        """
        Returns: (is_triggered, warning_message)
        """
        if not matches:
            return True, "Circuit Breaker Activated: No relevant source material discovered in the knowledge base."

        top = matches[0]
        if top.score < self.min_confidence_threshold:
            return (
                True,
                f"Circuit Breaker Activated: Retrieval confidence ({top.score:.3f}) below safety threshold "
                f"({self.min_confidence_threshold:.3f}). High risk of hallucination."
            )

        # Secondary, absolute gate: the fused hybrid score is an RRF rank-consensus
        # score, not a semantic similarity. On small corpora a single unrelated
        # chunk can trivially rank #1 across every channel and pass the threshold
        # above with a near-perfect score. Cross-check against the raw dense
        # cosine similarity (a real absolute semantic signal) whenever the dense
        # channel actually ran for this match (dense_score == 0.0 means either no
        # query vector was available, or this match came from a channel that
        # doesn't populate it — in that case we defer to the check above).
        if top.channel == "hybrid" and top.dense_score != 0.0 and top.dense_score < self.min_semantic_similarity:
            return (
                True,
                f"Circuit Breaker Activated: Rank-consensus score ({top.score:.3f}) cleared the safety threshold, "
                f"but raw semantic similarity ({top.dense_score:.3f}) is below the absolute floor "
                f"({self.min_semantic_similarity:.3f}). Likely a small-corpus false consensus rather than a "
                "genuine match."
            )

        return False, ""
