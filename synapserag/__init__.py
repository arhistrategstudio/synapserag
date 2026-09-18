"""
SynapseRAG — The Next-Generation Embeddable Tri-Brain RAG Engine.
"""

from .config import SynapseConfig
from .types import (
    Chunk,
    ChunkType,
    CitationSpan,
    DocumentSource,
    GraphEdge,
    GraphNode,
    QueryResult,
    RetrievalMatch,
    RetrievalMode,
)
from .engine import SynapseEngine

__version__ = "0.1.0"

__all__ = [
    "SynapseEngine",
    "SynapseConfig",
    "RetrievalMode",
    "QueryResult",
    "RetrievalMatch",
    "CitationSpan",
    "DocumentSource",
    "Chunk",
    "ChunkType",
    "GraphNode",
    "GraphEdge",
]
