"""
Regression tests for the 0.2.0 fixes:
1. script-aware tokenization (Cyrillic / diacritics),
2. exact chunk offsets and line numbers,
3. compact incremental SQLite persistence (+ legacy JSON migration),
4. circuit-breaker floor selection.

The legal-looking sentences below are short synthetic TEST FIXTURES written for these
tests; they are not quotations of real legislation or court decisions.
"""
import json
import os
import tempfile

import pytest

from synapserag import SynapseEngine, SynapseConfig
from synapserag.text import normalize_text, word_tokens
from synapserag.ingest.chunker import ContextualLateChunker
from synapserag.ingest.graph_extractor import FastGraphExtractor
from synapserag.storage import vector_store as vs_module
from synapserag.storage.vector_store import EmbeddedVectorStore
from synapserag.storage.sparse_store import EmbeddedSparseStore
from synapserag.storage.sqlite_db import DB_FILENAME
from synapserag.types import Chunk, DocumentSource


def _hash_config(tmpdir, **kw):
    return SynapseConfig(storage_dir=tmpdir, embedding_backend="hash", **kw)


# ---------------------------------------------------------------- 1. text normalization

def test_normalize_cyrillic_latin_and_diacritics_meet():
    assert normalize_text("Крађа") == normalize_text("Krađa") == "kradja"
    assert normalize_text("ЉУБАВ Њива Џак Ћup Шума Жито") == "ljubav njiva dzak cup suma zito"
    assert normalize_text("Café Über") == "cafe uber"


def test_word_tokens_keep_cyrillic_and_code_identifiers():
    toks = word_tokens("Члан 203: krađa; __init__ get_user-id")
    assert toks == ["clan", "203", "kradja", "__init__", "get_user-id"]


def test_sparse_tokenizer_no_longer_drops_cyrillic():
    assert EmbeddedSparseStore.tokenize("Тешка крађа") == ["teska", "kradja"]


def test_latin_query_finds_cyrillic_document():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = SynapseEngine(config=_hash_config(tmpdir))
        engine.ingest_text("Тестни документ о крађи бицикла у подруму.", uri="fixtures/cyr.txt")
        engine.ingest_text("Unrelated fixture about bakery inventory and flour.", uri="fixtures/other.txt")
        res = engine.query("krađa bicikla", mode="lexical", top_k=1)
        assert res.matches and res.citations[0].uri == "fixtures/cyr.txt"


def test_graph_extracts_non_ascii_capitalized_entities():
    phrases = FastGraphExtractor._capitalized_phrases("Testni Sud i Врховни суд; ColBERT model.")
    assert "Testni Sud" in phrases and "Врховни" in phrases and "ColBERT" in phrases


# ---------------------------------------------------------------- 2. offsets & lines

def test_sliding_window_chunks_have_exact_offsets_and_lines():
    words = " ".join(f"w{i}" for i in range(400))
    # wrap the long paragraph over many lines so line resolution is non-trivial
    body = "\n".join(words[i:i + 40] for i in range(0, len(words), 40))
    text = "naslov\n\n" + body
    chunks = ContextualLateChunker(chunk_size_words=120, overlap_words=20).create_chunks(
        DocumentSource(title="t"), text
    )
    lines = text.split("\n")
    children = chunks[1:]
    assert len(children) > 3
    for c in children:
        assert text[c.start_char:c.end_char] == c.content
        assert c.content.split()[0] in lines[c.start_line - 1]
        assert c.content.split()[-1] in lines[c.end_line - 1]
    # the last window must reach the end of the paragraph, and must not be a duplicate tail
    assert children[-1].end_char == len(text)


def test_line_numbers_ignore_form_feeds():
    text = "strana1\fjos teksta\n\nDrugi blok\nred dva\n\n\nTreci blok"
    chunks = ContextualLateChunker().create_chunks(DocumentSource(title="t"), text)
    starts = {c.content.split()[0]: c.start_line for c in chunks[1:]}
    assert starts == {"strana1": 1, "Drugi": 3, "Treci": 7}


# ---------------------------------------------------------------- 3. SQLite persistence

def test_persistence_uses_sqlite_and_is_incremental():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = SynapseEngine(config=_hash_config(tmpdir))
        engine.ingest_text("Prvi testni dokument o pritvoru.", uri="fixtures/a.txt")
        db = os.path.join(tmpdir, DB_FILENAME)
        assert os.path.exists(db)
        assert not os.path.exists(os.path.join(tmpdir, "vector_store.json"))
        # dirty sets are cleared after persist -> nothing is rewritten on the next persist
        assert not engine.vector_store._dirty and not engine.sparse_store._dirty
        engine.ingest_text("Drugi testni dokument o jemstvu.", uri="fixtures/b.txt")

        reloaded = SynapseEngine(config=_hash_config(tmpdir))
        assert len(reloaded.vector_store.chunks) == len(engine.vector_store.chunks)
        assert reloaded.sparse_store.num_docs == engine.sparse_store.num_docs
        assert len(reloaded.graph_store.nodes) == len(engine.graph_store.nodes)
        assert reloaded.query("jemstvu", mode="lexical").citations[0].uri == "fixtures/b.txt"


def test_token_embeddings_stored_compactly_and_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EmbeddedVectorStore(storage_dir=tmpdir)
        toks = [[0.5, -0.25, 0.125], [1.0, 0.0, -1.0]]
        store.add_chunk(Chunk(chunk_id="c1", content="x", dense_vector=[0.1, 0.2, 0.3], token_embeddings=toks))
        assert isinstance(store.token_embeddings["c1"], bytes)
        assert len(store.token_embeddings["c1"]) == 6 * 2  # float16
        store.persist()
        again = EmbeddedVectorStore(storage_dir=tmpdir)
        assert again.get_token_embeddings("c1") == toks  # exactly representable in float16
        assert again.get_dense_vector("c1") == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)


def test_pure_python_paths_without_numpy(monkeypatch):
    monkeypatch.setattr(vs_module, "HAS_NUMPY", False)
    store = EmbeddedVectorStore()
    store.add_chunk(Chunk(chunk_id="a", content="a", dense_vector=[1.0, 0.0], token_embeddings=[[1.0, 0.0]]))
    store.add_chunk(Chunk(chunk_id="b", content="b", dense_vector=[0.0, 1.0], token_embeddings=[[0.0, 1.0]]))
    assert store.search_dense([0.9, 0.1])[0][0] == "a"
    assert store.search_late_interaction([[0.0, 1.0]])[0][0] == "b"


def test_legacy_json_store_is_migrated_to_sqlite():
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy_chunk = {
            "chunk_id": "old1", "doc_id": "d1", "content": "stari zapis o kauciji",
            "contextualized_content": "stari zapis o kauciji", "start_char": 0, "end_char": 21,
            "start_line": 1, "end_line": 1, "dense_vector": [1.0, 0.0, 0.0],
            "token_embeddings": None, "entities": [], "metadata": {"uri": "legacy.txt"},
        }
        with open(os.path.join(tmpdir, "vector_store.json"), "w", encoding="utf-8") as f:
            json.dump({"old1": legacy_chunk}, f)
        with open(os.path.join(tmpdir, "sparse_store.json"), "w", encoding="utf-8") as f:
            json.dump({"num_docs": 1, "avg_doc_len": 4, "doc_lengths": {"old1": 4},
                       "inverted_index": {"stari": {"old1": 1}, "kauciji": {"old1": 1}}}, f)
        with open(os.path.join(tmpdir, "graph_store.json"), "w", encoding="utf-8") as f:
            json.dump({"nodes": [{"node_id": "node_1", "name": "Kaucija", "node_type": "entity",
                                  "associated_chunks": ["old1"], "metadata": {}}], "edges": []}, f)

        engine = SynapseEngine(config=_hash_config(tmpdir))
        assert engine.vector_store.get_chunk("old1") is not None
        engine.persist()
        assert os.path.exists(os.path.join(tmpdir, DB_FILENAME))

        # A fresh engine now loads from SQLite (legacy JSON is no longer needed).
        for name in ("vector_store.json", "sparse_store.json", "graph_store.json"):
            os.remove(os.path.join(tmpdir, name))
        migrated = SynapseEngine(config=_hash_config(tmpdir))
        assert migrated.vector_store.get_chunk("old1").content == "stari zapis o kauciji"
        assert migrated.sparse_store.search("kauciji")[0][0] == "old1"
        assert "node_1" in migrated.graph_store.nodes


# ---------------------------------------------------------------- 4. circuit-breaker floor

def test_semantic_floor_resolution():
    cfg = SynapseConfig()
    assert cfg.resolve_semantic_floor(neural_backend=False, late_interaction=True) == 0.22
    assert cfg.resolve_semantic_floor(neural_backend=True, late_interaction=False) == 0.40
    assert cfg.resolve_semantic_floor(neural_backend=True, late_interaction=True) == 0.60
    assert SynapseConfig(min_semantic_similarity=0.5).resolve_semantic_floor(True, True) == 0.5
