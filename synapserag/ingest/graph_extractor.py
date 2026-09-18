"""
Entity and Relationship Extractor.
Extracts named entities, code definitions, and concept relations from chunks to populate the Neuro-Associative Graph.
"""

from __future__ import annotations
from typing import List, Tuple
import re

from ..types import Chunk, GraphEdge, GraphNode
from ..storage.graph_store import EmbeddedGraphStore


class FastGraphExtractor:
    """
    Lightweight heuristic and syntactic entity extractor.
    Extracts entities (Classes, Functions, Modules, Concepts, Capitalized entities)
    and generates co-occurrence and hierarchical relations without requiring slow external LLM inference.
    """
    def __init__(self, graph_store: EmbeddedGraphStore):
        self.graph_store = graph_store

    def extract_and_link(self, chunk: Chunk) -> List[str]:
        """
        Extracts entities from chunk content, registers nodes in graph_store,
        links them to the chunk, and creates edges between co-occurring entities.
        """
        content = chunk.content
        extracted_entities = self._extract_entities(content)
        chunk.entities = extracted_entities

        node_ids: List[str] = []
        for ent in extracted_entities:
            nid = self.graph_store.link_entity_to_chunk(ent, chunk.chunk_id)
            node_ids.append(nid)

        # Build co-occurrence relational edges between entities in the same chunk
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                src = node_ids[i]
                dst = node_ids[j]
                edge = GraphEdge(
                    source_id=src,
                    target_id=dst,
                    relation="co_occurs_with",
                    weight=1.0,
                    source_chunk_id=chunk.chunk_id
                )
                self.graph_store.add_edge(edge)

        return extracted_entities

    def _extract_entities(self, text: str) -> List[str]:
        entities = set()

        # 1. Code symbols: class DefName, def func_name, function funcName
        code_defs = re.findall(r"(?:class|def|function|const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)", text)
        for c in code_defs:
            if len(c) > 2:
                entities.add(c)

        # 2. Capitalized phrases & acronyms (Named Entities: e.g. "Personalized PageRank", "ColBERT", "GPU")
        named_ents = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\b", text)
        for ne in named_ents:
            clean = ne.strip()
            # Filter out common start-of-sentence words
            if len(clean) > 2 and clean.lower() not in {"the", "this", "that", "there", "these", "those", "when", "what", "where"}:
                entities.add(clean)

        # 3. Quoted identifiers or backticked symbols (e.g. `vector_store`, "api_key")
        quoted = re.findall(r"`([a-zA-Z0-9_\-\.]{3,})`", text)
        for q in quoted:
            entities.add(q)

        return list(entities)[:20]  # Cap per chunk to avoid dense cliques
