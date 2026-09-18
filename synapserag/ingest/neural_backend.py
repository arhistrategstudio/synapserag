"""
Optional neural embedding backend backed by sentence-transformers.

SynapseRAG's storage/retrieval core has zero mandatory external dependencies (the
deterministic hash-based projection in MultiModalEmbedder always works). This module
is a pluggable upgrade: when sentence-transformers (and its cached model weights) are
available, the engine wires this backend in automatically for real semantic embeddings.
When it isn't, MultiModalEmbedder silently falls back to the hash-based projection.
"""

from __future__ import annotations
from typing import List, Optional


class SentenceTransformerBackend:
    """
    Lazy-loading wrapper around a sentence-transformers model.

    Provides both pooled dense embeddings (for standard semantic search) and
    per-token embeddings pulled from the underlying transformer's last hidden
    state (for ColBERT-style late-interaction MaxSim matching).
    """

    # Process-wide cache so multiple SynapseEngine instances (e.g. across tests,
    # or multiple engines in one host process) reuse already-loaded model weights
    # instead of reloading them from disk/network on every instantiation.
    _MODEL_CACHE: dict = {}

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_error: Optional[BaseException] = None
        self._attempted = False

    @property
    def is_available(self) -> bool:
        if not self._attempted:
            self._load()
        return self._model is not None

    @property
    def output_dim(self) -> Optional[int]:
        if not self.is_available:
            return None
        return self._model.get_sentence_embedding_dimension()

    def _load(self) -> None:
        self._attempted = True
        cache_key = (self.model_name, self.device)
        if cache_key in self._MODEL_CACHE:
            self._model = self._MODEL_CACHE[cache_key]
            return
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name, device=self.device)
            self._MODEL_CACHE[cache_key] = model
            self._model = model
        except Exception as exc:
            # Covers: library not installed, model weights not cached and no network,
            # incompatible torch build, etc. The engine falls back to the hash embedder.
            self._load_error = exc
            self._model = None

    def embed_text(self, text: str) -> List[float]:
        if not self.is_available:
            raise RuntimeError(f"SentenceTransformer backend unavailable: {self._load_error}")
        vector = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()

    def embed_tokens(self, text: str, max_tokens: int = 128) -> List[List[float]]:
        if not self.is_available:
            raise RuntimeError(f"SentenceTransformer backend unavailable: {self._load_error}")

        import torch

        transformer_module = self._model[0]
        tokenizer = transformer_module.tokenizer
        auto_model = transformer_module.auto_model

        features = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens + 2
        )
        with torch.no_grad():
            output = auto_model(**features)

        token_embeddings = output.last_hidden_state[0]
        attention_mask = features["attention_mask"][0]
        input_ids = features["input_ids"][0]
        special_ids = set(tokenizer.all_special_ids)

        vectors: List[List[float]] = []
        for idx in range(token_embeddings.shape[0]):
            if attention_mask[idx].item() == 0:
                continue
            if int(input_ids[idx].item()) in special_ids:
                continue
            vec = token_embeddings[idx]
            norm = vec.norm()
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())

        if not vectors:
            return [self.embed_text(text)]
        return vectors
