"""
Embedded Vector Store with Dense Embeddings and ColBERT Late-Interaction MaxSim Scoring.
Zero external server dependencies. Persists locally in the host app (SQLite, see sqlite_db.py).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import json
import math
from array import array
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..types import Chunk
from . import sqlite_db


class EmbeddedVectorStore:
    """
    In-process vector repository.
    Supports both single-vector dense representations and multi-vector ColBERT token representations.

    Memory layout (changed in 0.2.0 to scale beyond toy corpora):
    * ``dense_vectors[chunk_id]`` is a compact ``array('f')`` (float32), not a list of Python floats.
    * ``token_embeddings[chunk_id]`` is packed little-endian float16 bytes (dimension in
      ``token_dims[chunk_id]``); it is decoded on demand during late-interaction search.
    * Vectors are moved out of the stored ``Chunk`` objects (``chunk.dense_vector`` /
      ``chunk.token_embeddings`` are set to ``None`` after ``add_chunk``); use
      ``get_dense_vector`` / ``get_token_embeddings`` to read them back.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.chunks: Dict[str, Chunk] = {}
        self.dense_vectors: Dict[str, array] = {}
        self.token_embeddings: Dict[str, bytes] = {}
        self.token_dims: Dict[str, int] = {}
        self._dirty: Set[str] = set()
        self._dense_cache = None  # (chunk_ids, normalized numpy matrix) built lazily

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------ write
    def add_chunk(self, chunk: Chunk) -> None:
        """Store chunk and register dense and token-level embeddings."""
        cid = chunk.chunk_id
        self.chunks[cid] = chunk
        if chunk.dense_vector is not None:
            self.dense_vectors[cid] = array("f", chunk.dense_vector)
            self._dense_cache = None
        if chunk.token_embeddings:
            self.token_embeddings[cid] = sqlite_db.pack_f16_matrix(chunk.token_embeddings)
            self.token_dims[cid] = len(chunk.token_embeddings[0])
        # Vectors live in the compact stores above; drop the bulky Python lists.
        chunk.dense_vector = None
        chunk.token_embeddings = None
        self._dirty.add(cid)

    # ------------------------------------------------------------------ read
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def get_dense_vector(self, chunk_id: str) -> Optional[List[float]]:
        v = self.dense_vectors.get(chunk_id)
        return list(v) if v is not None else None

    def get_token_embeddings(self, chunk_id: str) -> Optional[List[List[float]]]:
        return sqlite_db.unpack_f16_matrix(self.token_embeddings.get(chunk_id), self.token_dims.get(chunk_id))

    # ------------------------------------------------------------------ dense search
    def search_dense(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Dense cosine similarity search across all indexed chunks."""
        if not self.dense_vectors or not query_vector:
            return []

        if HAS_NUMPY:
            return self._search_dense_numpy(query_vector, top_k)
        return self._search_dense_pure(query_vector, top_k)

    def _dense_matrix(self):
        if self._dense_cache is None:
            chunk_ids = list(self.dense_vectors.keys())
            matrix = np.array([self.dense_vectors[cid] for cid in chunk_ids], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self._dense_cache = (chunk_ids, matrix / norms)
        return self._dense_cache

    def _search_dense_numpy(self, query_vec: List[float], top_k: int) -> List[Tuple[str, float]]:
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        chunk_ids, matrix = self._dense_matrix()
        if matrix.shape[1] != q.shape[0]:
            return []
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

    # ------------------------------------------------------------------ late interaction
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

            for cid, blob in self.token_embeddings.items():
                dim = self.token_dims.get(cid)
                if not blob or dim != q_tokens.shape[1]:
                    continue
                d_tokens = np.frombuffer(blob, dtype="<f2").astype(np.float32).reshape(-1, dim)
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
            for cid in self.token_embeddings:
                d_tokens = self.get_token_embeddings(cid)
                if not d_tokens or len(d_tokens[0]) != len(query_token_embeddings[0]):
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

    # ------------------------------------------------------------------ persistence
    def persist(self) -> None:
        """Write chunks added since the last persist() to <storage_dir>/synapse.db."""
        if not self.storage_dir or not self._dirty:
            return
        rows = []
        for cid in self._dirty:
            c = self.chunks.get(cid)
            if c is None:
                continue
            dense = self.dense_vectors.get(cid)
            rows.append((
                c.chunk_id, c.doc_id, c.content, c.contextualized_content,
                c.start_char, c.end_char, c.start_line, c.end_line,
                json.dumps(c.entities, ensure_ascii=False), json.dumps(c.metadata, ensure_ascii=False),
                dense.tobytes() if dense is not None else None,
                self.token_embeddings.get(cid), self.token_dims.get(cid),
            ))
        con = sqlite_db.connect(self.storage_dir)
        try:
            with con:
                con.executemany(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
                )
        finally:
            con.close()
        self._dirty.clear()

    def _load(self) -> None:
        """Load index from synapse.db, or migrate a legacy vector_store.json."""
        if not self.storage_dir:
            return
        if sqlite_db.db_path(self.storage_dir).exists():
            con = sqlite_db.connect(self.storage_dir)
            try:
                for row in con.execute(
                    "SELECT chunk_id, doc_id, content, contextualized_content, start_char, end_char, "
                    "start_line, end_line, entities, metadata, dense, tokens, token_dim FROM chunks"
                ):
                    (cid, doc_id, content, ctx, sc, ec, sl, el, ents, meta, dense, tokens, tdim) = row
                    self.chunks[cid] = Chunk(
                        chunk_id=cid, doc_id=doc_id, content=content, contextualized_content=ctx,
                        start_char=sc, end_char=ec, start_line=sl, end_line=el,
                        entities=json.loads(ents or "[]"), metadata=json.loads(meta or "{}"),
                    )
                    if dense is not None:
                        self.dense_vectors[cid] = sqlite_db.unpack_f32(dense)
                    if tokens:
                        self.token_embeddings[cid] = tokens
                        self.token_dims[cid] = tdim
            finally:
                con.close()
            if self.chunks:
                return
        self._load_legacy_json()

    def _load_legacy_json(self) -> None:
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
                self.add_chunk(chunk)  # marks dirty -> migrated to SQLite on next persist()
        except Exception:
            pass
