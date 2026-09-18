"""
SynapseEngine — Master Engine Facade.
Provides the primary high-level API for ingestion, querying, and multi-brain coordination.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

from .types import Chunk, DocumentSource, QueryResult, RetrievalMode
from .config import SynapseConfig
from .storage.vector_store import EmbeddedVectorStore
from .storage.graph_store import EmbeddedGraphStore
from .storage.sparse_store import EmbeddedSparseStore
from .ingest.embedder import MultiModalEmbedder
from .ingest.neural_backend import SentenceTransformerBackend
from .ingest.chunker import ContextualLateChunker
from .ingest.graph_extractor import FastGraphExtractor
from .retrieval.engine import TriBrainRetriever
from .query.clue_engine import MemoClueEngine
from .query.router import AdaptiveQueryRouter
from .verify.citations import CitationEngine
from .verify.circuit_breaker import HallucinationCircuitBreaker


class SynapseEngine:
    """
    Main entry point for SynapseRAG.
    Embeddable in any Python process, desktop app, or server.
    """
    def __init__(self, config: Optional[SynapseConfig] = None):
        self.config = config or SynapseConfig()
        storage_path = str(self.config.get_storage_path())

        # Storage layer (Tri-Brain stores)
        self.vector_store = EmbeddedVectorStore(storage_dir=storage_path)
        self.graph_store = EmbeddedGraphStore(storage_dir=storage_path)
        self.sparse_store = EmbeddedSparseStore(storage_dir=storage_path)

        # Ingestion layer
        self.embedding_backend: Optional[SentenceTransformerBackend] = None
        self.embedder = self._build_embedder()
        self.chunker = ContextualLateChunker()
        self.graph_extractor = FastGraphExtractor(graph_store=self.graph_store)

        # Retrieval & Reasoning layer
        self.retriever = TriBrainRetriever(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            sparse_store=self.sparse_store,
            config=self.config
        )
        self.clue_engine = MemoClueEngine()
        self.router = AdaptiveQueryRouter()
        
        # Verification & Safety
        self.citation_engine = CitationEngine()
        self.circuit_breaker = HallucinationCircuitBreaker(
            min_confidence_threshold=self.config.min_confidence_threshold
        )

    def _build_embedder(self) -> MultiModalEmbedder:
        """
        Wires in the real sentence-transformers neural backend when available/requested,
        otherwise falls back to the zero-dependency deterministic hash embedder.
        """
        dense_fn = None
        tokens_fn = None

        if self.config.embedding_backend in ("auto", "sentence-transformers"):
            backend = SentenceTransformerBackend(model_name=self.config.embedding_model_name)
            if backend.is_available:
                self.embedding_backend = backend
                dense_fn = backend.embed_text
                tokens_fn = backend.embed_tokens
            elif self.config.embedding_backend == "sentence-transformers":
                raise RuntimeError(
                    f"embedding_backend='sentence-transformers' requested but could not load "
                    f"model '{self.config.embedding_model_name}': {backend._load_error}"
                )

        return MultiModalEmbedder(
            dim=self.config.embedding_dim,
            external_dense_fn=dense_fn,
            external_tokens_fn=tokens_fn
        )

    def ingest_text(
        self,
        text: str,
        uri: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentSource:
        """
        Ingest a text or code document into the engine.
        Processes through hierarchical chunker, generates embeddings, extracts entities into the graph,
        and indexes into sparse lexical store.
        """
        doc = DocumentSource(
            uri=uri,
            title=title or (Path(uri).name if uri else "Untitled"),
            metadata=metadata or {}
        )

        chunks = self.chunker.create_chunks(document=doc, raw_text=text)

        for chunk in chunks:
            # 1. Generate dense & ColBERT token embeddings
            chunk.dense_vector = self.embedder.embed_text(chunk.contextualized_content)
            if self.config.enable_late_chunking:
                chunk.token_embeddings = self.embedder.embed_tokens(chunk.contextualized_content)

            # 2. Extract entities and populate graph
            self.graph_extractor.extract_and_link(chunk)

            # 3. Add to stores
            self.vector_store.add_chunk(chunk)
            self.sparse_store.add_chunk(chunk)

        if self.config.persist_on_write:
            self.persist()

        return doc

    def ingest_file(self, file_path: Union[str, Path]) -> DocumentSource:
        """Read and ingest a single file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        return self.ingest_text(text=content, uri=str(path.resolve()), title=path.name)

    def ingest_directory(
        self,
        directory_path: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[DocumentSource]:
        """Batch ingest all matching files from a directory."""
        dir_p = Path(directory_path)
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        ext_set = {e.lower().lstrip(".") for e in extensions} if extensions else None
        docs = []

        glob_pattern = "**/*" if recursive else "*"
        for f in dir_p.glob(glob_pattern):
            if f.is_file():
                if ext_set is None or f.suffix.lower().lstrip(".") in ext_set:
                    try:
                        docs.append(self.ingest_file(f))
                    except Exception:
                        pass
        return docs

    def query(
        self,
        query: str,
        mode: Union[str, RetrievalMode] = RetrievalMode.AUTO,
        top_k: int = 5
    ) -> QueryResult:
        """
        Execute comprehensive retrieval with Tri-Brain fusion, speculative clues, and citations.
        """
        start_time = time.perf_counter()
        
        # Resolve retrieval mode
        if isinstance(mode, str):
            try:
                mode = RetrievalMode(mode.lower())
            except ValueError:
                mode = RetrievalMode.AUTO

        if mode == RetrievalMode.AUTO:
            effective_mode = self.router.route(query)
        else:
            effective_mode = mode

        # Step 1: System 1 speculative clues (MemoRAG pattern)
        clues = []
        if self.config.enable_clue_generation and effective_mode in (RetrievalMode.HYBRID, RetrievalMode.AUTO):
            clues = self.clue_engine.generate_clues(query, max_clues=self.config.max_speculative_clues)

        # Step 2: Extract keywords and embeddings
        query_vector = self.embedder.embed_text(query)
        query_token_embeddings = self.embedder.embed_tokens(query)
        keywords = self.sparse_store.tokenize(query)

        # Step 3: Tri-Brain Retrieval & Fusion
        matches = self.retriever.retrieve(
            query=query,
            query_vector=query_vector,
            query_token_embeddings=query_token_embeddings,
            keywords=keywords,
            mode=effective_mode,
            top_k=top_k
        )

        # Step 4: Verification and Circuit Breaker
        circuit_triggered = False
        warning_msg = None
        if self.config.enable_circuit_breaker:
            circuit_triggered, warning_msg = self.circuit_breaker.evaluate(matches)

        # Step 5: Deterministic Citations
        citations = self.citation_engine.generate_citations(matches)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return QueryResult(
            query=query,
            retrieval_mode=effective_mode,
            matches=matches,
            clues=clues,
            citations=citations,
            execution_time_ms=elapsed_ms,
            circuit_breaker_triggered=circuit_triggered,
            warning=warning_msg
        )

    def persist(self) -> None:
        """Flush all stores to disk."""
        self.vector_store.persist()
        self.graph_store.persist()
        self.sparse_store.persist()
