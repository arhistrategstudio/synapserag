"""
SynapseRAG Type Definitions & Domain Models.
Core schemas representing documents, chunks, graph structures, retrieval results, and citations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
import uuid


class RetrievalMode(str, Enum):
    HYBRID = "hybrid"                   # Tri-Brain: Dense + Graph + Sparse with dynamic RRF
    LATE_INTERACTION = "late_interaction" # ColBERT token-level MaxSim precision
    GRAPH_HOP = "graph_hop"             # Neuro-associative Personalized PageRank multi-hop
    LEXICAL = "lexical"                 # BM25 exact keyword matching
    AUTO = "auto"                       # Dynamically decided by QueryRouter


class ChunkType(str, Enum):
    PARENT = "parent"   # Structural section/document level
    CHILD = "child"     # Granular text/code segment with late-chunking tensors


@dataclass
class DocumentSource:
    """Represents the originating document or input stream."""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    uri: Optional[str] = None
    title: Optional[str] = None
    global_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class Chunk:
    """
    Granular text or code segment.
    Maintains hierarchical relationship and exact position for citations.
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    content: str = ""
    contextualized_content: str = ""  # Content prepended with global document context
    chunk_type: ChunkType = ChunkType.CHILD
    parent_id: Optional[str] = None
    
    # Exact positional spans for deterministic citation
    start_char: int = 0
    end_char: int = 0
    start_line: int = 1
    end_line: int = 1
    
    # Representations
    dense_vector: Optional[List[float]] = None
    token_embeddings: Optional[List[List[float]]] = None # For ColBERT Late Interaction
    sparse_tokens: Dict[str, float] = field(default_factory=dict)
    
    # Metadata & Entities
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """Entity or concept node in the neuro-associative knowledge graph."""
    node_id: str
    name: str
    node_type: str = "entity" # entity, concept, file, section
    embedding: Optional[List[float]] = None
    associated_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Semantic relation connecting two nodes in the graph."""
    source_id: str
    target_id: str
    relation: str = "relates_to"
    weight: float = 1.0
    vector_bias: float = 1.0 # Modulated dynamically by vector similarity
    source_chunk_id: Optional[str] = None


@dataclass
class CitationSpan:
    """Verifiable source reference grounded to exact lines and characters."""
    doc_id: str
    uri: Optional[str]
    chunk_id: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    exact_quote: str
    confidence_score: float


@dataclass
class RetrievalMatch:
    """Individual retrieval candidate score across retrieval channels."""
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    graph_score: float = 0.0
    sparse_score: float = 0.0
    channel: str = "hybrid"
    explanation: Optional[str] = None


@dataclass
class QueryResult:
    """Comprehensive retrieval outcome with multi-hop grounding and citations."""
    query: str
    retrieval_mode: RetrievalMode
    matches: List[RetrievalMatch] = field(default_factory=list)
    clues: List[str] = field(default_factory=list) # Speculative clues from System 1
    citations: List[CitationSpan] = field(default_factory=list)
    execution_time_ms: float = 0.0
    circuit_breaker_triggered: bool = False
    warning: Optional[str] = None
