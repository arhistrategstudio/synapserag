"""
Deterministic Citation Engine.
Maps chunks to exact line numbers, character spans, and verifiable quotes.
"""

from __future__ import annotations
from typing import List

from ..types import CitationSpan, RetrievalMatch


class CitationEngine:
    """
    Constructs verifiable, grounded citation anchors for retrieved evidence.
    Enables IDEs, agents, and end users to jump directly to exact file, line, and character offsets.
    """
    @staticmethod
    def generate_citations(matches: List[RetrievalMatch]) -> List[CitationSpan]:
        citations: List[CitationSpan] = []
        for match in matches:
            c = match.chunk
            quote = c.content[:200].strip() + ("..." if len(c.content) > 200 else "")
            citation = CitationSpan(
                doc_id=c.doc_id,
                uri=c.metadata.get("uri"),
                chunk_id=c.chunk_id,
                start_line=c.start_line,
                end_line=c.end_line,
                start_char=c.start_char,
                end_char=c.end_char,
                exact_quote=quote,
                confidence_score=match.score
            )
            citations.append(citation)
        return citations
