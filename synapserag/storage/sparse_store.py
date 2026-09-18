"""
Embedded Sparse Lexical Index (Okapi BM25).
High-precision keyword and code symbol retrieval with zero external dependencies.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

from ..types import Chunk


class EmbeddedSparseStore:
    """
    Inverted index implementing Okapi BM25 scoring.
    Tuned for natural language text as well as code tokens (identifiers, methods, camelCase).
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, storage_dir: Optional[str] = None):
        self.k1 = k1
        self.b = b
        self.storage_dir = Path(storage_dir) if storage_dir else None
        
        self.doc_lengths: Dict[str, int] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        # Inverted index: term -> {chunk_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.avg_doc_len: float = 0.0
        self.num_docs: int = 0
        
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Code-aware tokenizer splitting on punctuation, snake_case, and camelCase."""
        if not text:
            return []
        # Split camelCase words (e.g., getUserProfile -> getUser Profile)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Extract alphanumeric tokens and identifiers
        raw_tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
        tokens = []
        for t in raw_tokens:
            if len(t) > 1:
                tokens.append(t)
            if "_" in t or "-" in t:
                subwords = re.split(r"[_\-]+", t)
                for sw in subwords:
                    if len(sw) > 1:
                        tokens.append(sw)
        return tokens

    def add_chunk(self, chunk: Chunk) -> None:
        """Index chunk tokens into the inverted index."""
        content = chunk.contextualized_content or chunk.content
        tokens = self.tokenize(content)
        cid = chunk.chunk_id
        
        self.doc_tokens[cid] = tokens
        self.doc_lengths[cid] = len(tokens)
        
        tf_counts = Counter(tokens)
        for term, freq in tf_counts.items():
            self.inverted_index[term][cid] = freq
            
        self.num_docs = len(self.doc_lengths)
        self.avg_doc_len = sum(self.doc_lengths.values()) / max(1, self.num_docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Compute BM25 scores for the given query against all indexed chunks."""
        query_tokens = self.tokenize(query)
        if not query_tokens or self.num_docs == 0:
            return []

        scores: Dict[str, float] = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            posting = self.inverted_index[term]
            doc_freq = len(posting)
            # Standard BM25 IDF formula
            idf = math.log((self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            idf = max(0.0, idf)

            for cid, tf in posting.items():
                d_len = self.doc_lengths.get(cid, self.avg_doc_len)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(1.0, self.avg_doc_len)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / max(1e-10, denom))
                scores[cid] += term_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def persist(self) -> None:
        """Persist inverted index to disk."""
        if not self.storage_dir:
            return
        data = {
            "num_docs": self.num_docs,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "inverted_index": {t: dict(p) for t, p in self.inverted_index.items()}
        }
        with open(self.storage_dir / "sparse_store.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load inverted index from disk."""
        if not self.storage_dir:
            return
        path = self.storage_dir / "sparse_store.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.num_docs = data.get("num_docs", 0)
            self.avg_doc_len = data.get("avg_doc_len", 0.0)
            self.doc_lengths = data.get("doc_lengths", {})
            self.inverted_index = defaultdict(dict, {
                t: {cid: freq for cid, freq in p.items()}
                for t, p in data.get("inverted_index", {}).items()
            })
        except Exception:
            pass
