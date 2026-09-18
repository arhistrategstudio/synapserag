import tempfile
import shutil
from pathlib import Path
from synapserag.types import Chunk, GraphNode, GraphEdge
from synapserag.storage.vector_store import EmbeddedVectorStore
from synapserag.storage.graph_store import EmbeddedGraphStore
from synapserag.storage.sparse_store import EmbeddedSparseStore


def test_vector_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EmbeddedVectorStore(storage_dir=tmpdir)
        c1 = Chunk(chunk_id="c1", content="Machine learning models", dense_vector=[1.0, 0.0, 0.0])
        c2 = Chunk(chunk_id="c2", content="Database query optimization", dense_vector=[0.0, 1.0, 0.0])
        store.add_chunk(c1)
        store.add_chunk(c2)

        results = store.search_dense([0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == "c1"

        # Test persistence
        store.persist()
        store2 = EmbeddedVectorStore(storage_dir=tmpdir)
        assert store2.get_chunk("c1") is not None


def test_late_interaction_maxsim():
    store = EmbeddedVectorStore()
    # Doc 1 has tokens [[1,0], [0,1]]
    c1 = Chunk(chunk_id="c1", content="doc1", token_embeddings=[[1.0, 0.0], [0.0, 1.0]])
    # Doc 2 has tokens [[0,1], [0,1]]
    c2 = Chunk(chunk_id="c2", content="doc2", token_embeddings=[[0.0, 1.0], [0.0, 1.0]])
    store.add_chunk(c1)
    store.add_chunk(c2)

    # Query tokens [[1,0]]
    scores = store.search_late_interaction(query_token_embeddings=[[1.0, 0.0]], top_k=2)
    assert len(scores) == 2
    assert scores[0][0] == "c1"


def test_graph_store_and_ppr():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = EmbeddedGraphStore(storage_dir=tmpdir)
        n1 = GraphNode(node_id="n1", name="Transformer", associated_chunks=["c1"])
        n2 = GraphNode(node_id="n2", name="Attention", associated_chunks=["c2"])
        n3 = GraphNode(node_id="n3", name="BERT", associated_chunks=["c3"])
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        graph.add_edge(GraphEdge(source_id="n1", target_id="n2", weight=1.0))
        graph.add_edge(GraphEdge(source_id="n2", target_id="n3", weight=1.0))

        # PPR starting from Transformer
        ppr_scores = graph.compute_vector_biased_ppr(seed_nodes=["n1"])
        assert "n1" in ppr_scores
        assert "n2" in ppr_scores
        assert "n3" in ppr_scores
        # n2 should have higher score than n3 (1 hop vs 2 hops)
        assert ppr_scores["n2"] > ppr_scores["n3"]

        chunk_scores = graph.get_chunk_scores_from_ppr(ppr_scores)
        assert len(chunk_scores) == 3


def test_bm25_sparse_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        sparse = EmbeddedSparseStore(storage_dir=tmpdir)
        c1 = Chunk(chunk_id="c1", content="def calculate_personalized_pagerank(nodes): pass")
        c2 = Chunk(chunk_id="c2", content="def render_user_interface(props): pass")
        sparse.add_chunk(c1)
        sparse.add_chunk(c2)

        res = sparse.search("pagerank", top_k=2)
        assert len(res) == 1
        assert res[0][0] == "c1"
