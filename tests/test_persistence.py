import tempfile

from synapserag import SynapseEngine, SynapseConfig


def _fast_config(storage_dir: str, **overrides) -> SynapseConfig:
    # Force the deterministic hash embedder so this test is fast and self-contained,
    # independent of whether sentence-transformers/model weights are available.
    return SynapseConfig(storage_dir=storage_dir, embedding_backend="hash", **overrides)


def test_engine_persist_and_reload_from_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_text = """
        SynapseRAG Persistence Layer:
        The engine writes vector, graph, and sparse indices to disk after every ingest
        so a new engine instance pointed at the same storage_dir can resume immediately.
        """

        # First engine: ingest and persist (persist_on_write defaults to True).
        engine1 = SynapseEngine(config=_fast_config(tmpdir))
        doc = engine1.ingest_text(text=doc_text, uri="docs/persistence.md", title="Persistence Layer")
        result1 = engine1.query("How does the engine persist its indices?", mode="hybrid", top_k=3)
        assert len(result1.matches) > 0

        # Second engine: fresh instance, same storage_dir, no re-ingestion.
        engine2 = SynapseEngine(config=_fast_config(tmpdir))

        # Chunks/vectors survived the reload.
        assert len(engine2.vector_store.chunks) == len(engine1.vector_store.chunks)
        assert len(engine2.vector_store.chunks) > 0

        # Graph nodes (extracted entities) survived the reload.
        assert len(engine2.graph_store.nodes) == len(engine1.graph_store.nodes)

        # Sparse (BM25) index survived the reload.
        assert engine2.sparse_store.num_docs == engine1.sparse_store.num_docs
        assert engine2.sparse_store.num_docs > 0

        # Querying the reloaded engine returns matches from the original document
        # without needing to call ingest_text again.
        result2 = engine2.query("How does the engine persist its indices?", mode="hybrid", top_k=3)
        assert len(result2.matches) > 0
        assert result2.citations[0].uri == "docs/persistence.md"
        assert result2.matches[0].chunk.doc_id == doc.doc_id


def test_engine_no_persist_on_write_requires_explicit_persist():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine1 = SynapseEngine(config=_fast_config(tmpdir, persist_on_write=False))
        engine1.ingest_text(text="Ephemeral content never flushed to disk.", uri="docs/ephemeral.md")

        # Nothing was flushed yet, so a fresh engine sees an empty store.
        engine2 = SynapseEngine(config=_fast_config(tmpdir, persist_on_write=False))
        assert len(engine2.vector_store.chunks) == 0

        # Explicit persist() makes the data available to subsequent engine instances.
        engine1.persist()
        engine3 = SynapseEngine(config=_fast_config(tmpdir, persist_on_write=False))
        assert len(engine3.vector_store.chunks) == len(engine1.vector_store.chunks)
        assert len(engine3.vector_store.chunks) > 0
