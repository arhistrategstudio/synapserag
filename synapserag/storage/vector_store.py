"""
Embedded Vector Store with Dense Embeddings and ColBERT Late-Interaction MaxSim Scoring.
Zero external server dependencies. Persists locally in the host app.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import math
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..types import Chunk


class EmbeddedVectorStore:
    """
    In-process high performance vector repository.
    Supports both single-vector dense representations and multi-vector ColBERT token representations.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.chunks: Dict[str, Chunk] = {}
        self.dense_vectors: Dict[str, List[float]] = {}
        self.token_embeddings: Dict[str, List[List[float]]] = {}
        
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def add_chunk(self, chunk: Chunk) -> None:
        """Store chunk and register dense and token-level embeddings."""
        self.chunks[chunk.chunk_id] = chunk
        if chunk.dense_vector is not None:
            self.dense_vectors[chunk.chunk_id] = chunk.dense_vector
        if chunk.token_embeddings is not None:
            self.token_embeddings[chunk.chunk_id] = chunk.token_embeddings

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def search_dense(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Dense cosine similarity search across all indexed chunks."""
        if not self.dense_vectors or not query_vector:
            return []

        if HAS_NUMPY:
            return self._search_dense_numpy(query_vector, top_k)
        return self._search_dense_pure(query_vector, top_k)

    def _search_dense_numpy(self, query_vec: List[float], top_k: int) -> List[Tuple[str, float]]:
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        chunk_ids = list(self.dense_vectors.keys())
        matrix = np.array([self.dense_vectors[cid] for cid in chunk_ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        matrix = matrix / norms

        scores = np.dot(matrix, q)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(chunk_ids[idx], float(scores[idx])) for idx in top_indices]

    def _search_dense_pure(self, query_vec: List[float], top_k: int) -> List[Tuple[str, float]]:
        q_norm = math.sqrt(sum(x * x for x in query_vec))
        if q_norm == 0:
            return []

        results = []
        for cid, vec in self.dense_vectors.items():
            dot = sum(a * b for a, b in zip(query_vec, vec))
            v_norm = math.sqrt(sum(x * x for x in vec))
            sim = dot / (q_norm * v_norm) if v_norm > 0 else 0.0
            results.append((cid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_late_interaction(
        self, query_token_embeddings: List[List[float]], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        ColBERT MaxSim Late Interaction operator:
        Score(Q, D) = sum_{q_i in Q} max_{d_j in D} (cosine(q_i, d_j))
        Provides token-level surgical matching across long contexts.
        """
        if not self.token_embeddings or not query_token_embeddings:
            return []

        scores: List[Tuple[str, float]] = []

        if HAS_NUMPY:
            q_tokens = np.array(query_token_embeddings, dtype=np.float32)
            q_norms = np.linalg.norm(q_tokens, axis=1, keepdims=True)
            q_norms[q_norms == 0] = 1e-10
            q_tokens = q_tokens / q_norms

            for cid, d_tokens_list in self.token_embeddings.items():
                if not d_tokens_list:
                    continue
                d_tokens = np.array(d_tokens_list, dtype=np.float32)
                d_norms = np.linalg.norm(d_tokens, axis=1, keepdims=True)
                d_norms[d_norms == 0] = 1e-10
                d_tokens = d_tokens / d_norms

                # Dot product between all query tokens and document tokens: shape (len_Q, len_D)
                sim_matrix = np.matmul(q_tokens, d_tokens.T)
                # MaxSim: take maximum for each query token across all doc tokens, then sum
                max_sims = np.max(sim_matrix, axis=1)
                score = float(np.sum(max_sims)) / len(query_token_embeddings)
                scores.append((cid, score))
        else:
            for cid, d_tokens in self.token_embeddings.items():
                if not d_tokens:
                    continue
                total_sim = 0.0
                for q_tok in query_token_embeddings:
                    q_len = math.sqrt(sum(x * x for x in q_tok)) or 1e-10
                    max_d = -1.0
                    for d_tok in d_tokens:
                        d_len = math.sqrt(sum(x * x for x in d_tok)) or 1e-10
                        cos = sum(a * b for a, b in zip(q_tok, d_tok)) / (q_len * d_len)
                        if cos > max_d:
                            max_d = cos
                    total_sim += max_d
                scores.append((cid, total_sim / len(query_token_embeddings)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def persist(self) -> None:
        """Serialize index and chunks to local disk file."""
        if not self.storage_dir:
            return
        data_path = self.storage_dir / "vector_store.json"
        serialized_chunks = {
            cid: {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "contextualized_content": c.contextualized_content,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "dense_vector": c.dense_vector,
                "token_embeddings": c.token_embeddings,
                "entities": c.entities,
                "metadata": c.metadata,
            }
            for cid, c in self.chunks.items()
        }
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(serialized_chunks, f)

    def _load(self) -> None:
        """Load index from disk if present."""
        if not self.storage_dir:
            return
        data_path = self.storage_dir / "vector_store.json"
        if not data_path.exists():
            return
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cid, item in data.items():
                chunk = Chunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    content=item["content"],
                    contextualized_content=item.get("contextualized_content", item["content"]),
                    start_char=item.get("start_char", 0),
                    end_char=item.get("end_char", 0),
                    start_line=item.get("start_line", 1),
                    end_line=item.get("end_line", 1),
                    dense_vector=item.get("dense_vector"),
                    token_embeddings=item.get("token_embeddings"),
                    entities=item.get("entities", []),
                    metadata=item.get("metadata", {}),
                )
                self.add_chunk(chunk)
        except Exception:
            pass
