"""
Shared SQLite persistence helpers (standard library only).

All three stores persist into one file, ``<storage_dir>/synapse.db``. Compared with the
previous JSON files this gives:

* compact binary vectors (float32 dense vectors, float16 per-token vectors) instead of
  decimal text — roughly 6-12x smaller on disk;
* incremental writes: ``persist()`` only writes rows added since the last persist, instead
  of re-serializing the entire index after every ingested document;
* one file that is easy to back up or copy.

Legacy ``vector_store.json`` / ``graph_store.json`` / ``sparse_store.json`` files are still
read on startup when no ``synapse.db`` exists yet, and are migrated on the next persist().
"""

from __future__ import annotations

import sqlite3
import struct
from array import array
from pathlib import Path
from typing import List, Optional, Sequence

DB_FILENAME = "synapse.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    content TEXT,
    contextualized_content TEXT,
    start_char INTEGER, end_char INTEGER,
    start_line INTEGER, end_line INTEGER,
    entities TEXT,
    metadata TEXT,
    dense BLOB,
    tokens BLOB,
    token_dim INTEGER
);
CREATE TABLE IF NOT EXISTS sparse_docs (chunk_id TEXT PRIMARY KEY, length INTEGER);
CREATE TABLE IF NOT EXISTS sparse_postings (term TEXT, chunk_id TEXT, tf INTEGER, PRIMARY KEY (term, chunk_id));
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT PRIMARY KEY, name TEXT, node_type TEXT, associated_chunks TEXT, metadata TEXT
);
CREATE TABLE IF NOT EXISTS graph_edges (
    source_id TEXT, target_id TEXT, relation TEXT, weight REAL, source_chunk_id TEXT
);
"""


def db_path(storage_dir: Path) -> Path:
    return Path(storage_dir) / DB_FILENAME


def connect(storage_dir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path(storage_dir)))
    con.executescript(_SCHEMA)
    return con


# ---- vector (de)serialization -------------------------------------------------------

def pack_f32(vec: Optional[Sequence[float]]) -> Optional[bytes]:
    if vec is None:
        return None
    return array("f", vec).tobytes()


def unpack_f32(blob: Optional[bytes]) -> Optional[array]:
    if blob is None:
        return None
    a = array("f")
    a.frombytes(blob)
    return a


def pack_f16_matrix(rows: Optional[Sequence[Sequence[float]]]) -> Optional[bytes]:
    """Pack a list of equal-length vectors as little-endian float16 ('e')."""
    if not rows:
        return None
    dim = len(rows[0])
    flat: List[float] = [float(x) for r in rows for x in r]
    return struct.pack(f"<{len(flat)}e", *flat) if dim else None


def unpack_f16_matrix(blob: Optional[bytes], dim: Optional[int]) -> Optional[List[List[float]]]:
    if not blob or not dim:
        return None
    n = len(blob) // 2
    flat = struct.unpack(f"<{n}e", blob)
    return [list(flat[i:i + dim]) for i in range(0, n, dim)]
