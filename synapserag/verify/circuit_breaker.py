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
    def __init__(self, min_confidence_threshold: float = 0.05):
        self.min_confidence_threshold = min_confidence_threshold

    def evaluate(self, matches: List[RetrievalMatch]) -> Tuple[bool, str]:
        """
        Returns: (is_triggered, warning_message)
        """
        if not matches:
            return True, "Circuit Breaker Activated: No relevant source material discovered in the knowledge base."

        top_score = matches[0].score
        if top_score < self.min_confidence_threshold:
            return (
                True,
                f"Circuit Breaker Activated: Retrieval confidence ({top_score:.3f}) below safety threshold "
                f"({self.min_confidence_threshold:.3f}). High risk of hallucination."
            )

        return False, ""
