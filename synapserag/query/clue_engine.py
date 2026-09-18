"""
System 1 Memo-Clue Generator (Inspired by MemoRAG).
Generates speculative hypotheses and latent navigation clues for ambiguous or high-level queries.
"""

from __future__ import annotations
from typing import Callable, List, Optional
import re


class MemoClueEngine:
    """
    Generates speculative clues to bridge the semantic disconnect between
    abstract user intent and low-level source fragments.
    """
    def __init__(self, external_llm_fn: Optional[Callable[[str], List[str]]] = None):
        self.external_llm_fn = external_llm_fn

    def generate_clues(self, query: str, max_clues: int = 3) -> List[str]:
        """
        Produces speculative answers or contextual hints.
        Uses external LLM if configured, otherwise applies heuristic associative expansion.
        """
        if self.external_llm_fn:
            try:
                clues = self.external_llm_fn(query)
                if clues:
                    return clues[:max_clues]
            except Exception:
                pass

        return self._heuristic_clue_expansion(query, max_clues)

    def _heuristic_clue_expansion(self, query: str, max_clues: int) -> List[str]:
        clues = []
        clean = query.strip()

        # Check for error/exception inquiries
        if any(w in clean.lower() for w in ["error", "fail", "exception", "bug", "crash", "issue"]):
            clues.append("Look for error handling, try-catch blocks, fallback controllers, or circuit breakers.")

        # Check for configuration/setup inquiries
        if any(w in clean.lower() for w in ["config", "setup", "install", "environment", "setting"]):
            clues.append("Examine configuration parameters, environment variables, default settings, and constructors.")

        # Check for connection/integration inquiries
        if any(w in clean.lower() for w in ["connect", "agent", "mcp", "integration", "api", "tool"]):
            clues.append("Look for agent connectors, MCP server endpoints, tool calling interfaces, and protocol bindings.")

        # Extract primary noun phrases or keywords as a fallback clue
        words = [w for w in re.findall(r"[a-zA-Z0-9_\-]+", clean) if len(w) > 3]
        if words:
            clues.append(f"Trace relations and definitions around core symbols: {', '.join(words[:4])}")

        return clues[:max_clues]
