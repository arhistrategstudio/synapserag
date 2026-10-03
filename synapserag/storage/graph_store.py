"""
Embedded Neuro-Associative Knowledge Graph Store.
Implements Vector-Biased Personalized PageRank (PPR) for instant multi-hop associative retrieval.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import json
import math
from pathlib import Path
from collections import defaultdict

from ..types import GraphNode, GraphEdge
from ..text import normalize_text
from . import sqlite_db


class EmbeddedGraphStore:
    """
    Lightweight, embedded graph knowledge store.
    Represents semantic relationships between entities, concepts, files, and text chunks.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.nodes: Dict[str, GraphNode] = {}
        # Outgoing edges: source_id -> list of GraphEdge
        self.adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)
        # Inverted index: entity/concept name (lowercase) -> node_id
        self.name_to_node: Dict[str, str] = {}
        # Mapping from chunk_id to associated node_ids
        self.chunk_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        # Incremental persistence bookkeeping
        self._dirty_nodes: Set[str] = set()
        self._new_edges: List[GraphEdge] = []
        self._loading = False

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self.name_to_node[normalize_text(node.name).strip()] = node.node_id
        for cid in node.associated_chunks:
            self.chunk_to_nodes[cid].add(node.node_id)
        if not self._loading:
            self._dirty_nodes.add(node.node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        self.adjacency[edge.source_id].append(edge)
        if not self._loading and not edge.relation.startswith("rev_"):
            self._new_edges.append(edge)
        # Also add reciprocal back-edge for associative spreading
        reverse_edge = GraphEdge(
            source_id=edge.target_id,
            target_id=edge.source_id,
            relation=f"rev_{edge.relation}",
            weight=edge.weight * 0.8,
            source_chunk_id=edge.source_chunk_id
        )
        self.adjacency[edge.target_id].append(reverse_edge)

    def link_entity_to_chunk(self, entity_name: str, chunk_id: str, node_type: str = "entity") -> str:
        """Create or update entity node and link it to the chunk."""
        clean_name = normalize_text(entity_name).strip()
        if clean_name in self.name_to_node:
            nid = self.name_to_node[clean_name]
            node = self.nodes[nid]
            if chunk_id not in node.associated_chunks:
                node.associated_chunks.append(chunk_id)
                self._dirty_nodes.add(nid)
            self.chunk_to_nodes[chunk_id].add(nid)
            return nid
        else:
            nid = f"node_{len(self.nodes) + 1}"
            node = GraphNode(
                node_id=nid,
                name=entity_name,
                node_type=node_type,
                associated_chunks=[chunk_id]
            )
            self.add_node(node)
            return nid

    def find_nodes_by_name(self, text_keywords: List[str]) -> List[str]:
        """Find nodes whose names match any of the given keywords."""
        matched_ids = []
        for kw in text_keywords:
            clean = normalize_text(kw).strip()
            if clean in self.name_to_node:
                matched_ids.append(self.name_to_node[clean])
            else:
                for name, nid in self.name_to_node.items():
                    if clean in name or name in clean:
                        matched_ids.append(nid)
        return list(set(matched_ids))

    def compute_vector_biased_ppr(
        self,
        seed_nodes: List[str],
        query_vector: Optional[List[float]] = None,
        damping: float = 0.85,
        max_iter: int = 30,
        tol: float = 1e-6,
        vector_bias_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculates Vector-Biased Personalized PageRank (PPR).
        Transition probabilities are amplified when neighbor node embeddings align with the query.
        """
        all_nodes = list(self.nodes.keys())
        n = len(all_nodes)
        if n == 0 or not seed_nodes:
            return {}

        # Construct personalization vector p (uniform over seed nodes)
        valid_seeds = [s for s in seed_nodes if s in self.nodes]
        if not valid_seeds:
            return {}
            
        p = {node_id: (1.0 / len(valid_seeds) if node_id in valid_seeds else 0.0) for node_id in all_nodes}
        r = {node_id: p[node_id] for node_id in all_nodes}

        # Precompute transition matrix with vector bias
        # For each node u, compute normalized weights to neighbors
        transition_weights: Dict[str, Dict[str, float]] = {}
        for u in all_nodes:
            edges = self.adjacency.get(u, [])
            if not edges:
                continue
            
            raw_weights = {}
            for edge in edges:
                v = edge.target_id
                base_w = edge.weight
                
                # Apply vector bias if both query and target node have embeddings
                target_node = self.nodes.get(v)
                v_sim = 0.0
                if query_vector and target_node and target_node.embedding:
                    dot = sum(a * b for a, b in zip(query_vector, target_node.embedding))
                    v_sim = max(0.0, dot)
                
                effective_weight = base_w * (1.0 + vector_bias_weight * v_sim)
                raw_weights[v] = raw_weights.get(v, 0.0) + effective_weight

            total_w = sum(raw_weights.values())
            if total_w > 0:
                transition_weights[u] = {v: w / total_w for v, w in raw_weights.items()}

        # Power iteration
        for _ in range(max_iter):
            r_new = {node_id: (1.0 - damping) * p[node_id] for node_id in all_nodes}
            
            for u, prob in r.items():
                if u in transition_weights:
                    for v, trans_prob in transition_weights[u].items():
                        r_new[v] += damping * prob * trans_prob
                else:
                    # Dangling node distribution
                    for v in all_nodes:
                        r_new[v] += damping * prob * p[v]

            # Check convergence (L1 norm)
            diff = sum(abs(r_new[nid] - r[nid]) for nid in all_nodes)
            r = r_new
            if diff < tol:
                break

        return r

    def get_chunk_scores_from_ppr(self, node_ppr_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """Project graph node PPR activation scores back onto associated text chunks."""
        chunk_scores: Dict[str, float] = defaultdict(float)
        for node_id, score in node_ppr_scores.items():
            node = self.nodes.get(node_id)
            if node and node.associated_chunks:
                # Distribute score across chunks containing this entity
                chunk_weight = score / len(node.associated_chunks)
                for cid in node.associated_chunks:
                    chunk_scores[cid] += chunk_weight

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def persist(self) -> None:
        """Write nodes/edges changed since the last persist() to <storage_dir>/synapse.db."""
        if not self.storage_dir or (not self._dirty_nodes and not self._new_edges):
            return
        node_rows = []
        for nid in self._dirty_nodes:
            n = self.nodes.get(nid)
            if n is None:
                continue
            node_rows.append((n.node_id, n.name, n.node_type,
                              json.dumps(n.associated_chunks), json.dumps(n.metadata, ensure_ascii=False)))
        edge_rows = [(e.source_id, e.target_id, e.relation, e.weight, e.source_chunk_id)
                     for e in self._new_edges]
        con = sqlite_db.connect(self.storage_dir)
        try:
            with con:
                con.executemany("INSERT OR REPLACE INTO graph_nodes VALUES (?,?,?,?,?)", node_rows)
                con.executemany("INSERT INTO graph_edges VALUES (?,?,?,?,?)", edge_rows)
        finally:
            con.close()
        self._dirty_nodes.clear()
        self._new_edges.clear()

    def _load(self) -> None:
        """Load graph from synapse.db, or migrate a legacy graph_store.json."""
        if not self.storage_dir:
            return
        if sqlite_db.db_path(self.storage_dir).exists():
            con = sqlite_db.connect(self.storage_dir)
            self._loading = True
            try:
                for nid, name, ntype, chunks, meta in con.execute("SELECT * FROM graph_nodes"):
                    self.add_node(GraphNode(node_id=nid, name=name, node_type=ntype,
                                            associated_chunks=json.loads(chunks or "[]"),
                                            metadata=json.loads(meta or "{}")))
                for src, dst, rel, w, scid in con.execute("SELECT * FROM graph_edges"):
                    self.add_edge(GraphEdge(source_id=src, target_id=dst, relation=rel,
                                            weight=w, source_chunk_id=scid))
            finally:
                self._loading = False
                con.close()
            if self.nodes:
                return
        self._load_legacy_json()

    def _load_legacy_json(self) -> None:
        path = self.storage_dir / "graph_store.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Not wrapped in _loading: everything is marked dirty so the next persist()
            # migrates the legacy JSON graph into SQLite.
            for item in data.get("nodes", []):
                self.add_node(GraphNode(
                    node_id=item["node_id"],
                    name=item["name"],
                    node_type=item.get("node_type", "entity"),
                    associated_chunks=item.get("associated_chunks", []),
                    metadata=item.get("metadata", {})
                ))
            for item in data.get("edges", []):
                self.add_edge(GraphEdge(
                    source_id=item["source_id"],
                    target_id=item["target_id"],
                    relation=item.get("relation", "relates_to"),
                    weight=item.get("weight", 1.0),
                    source_chunk_id=item.get("source_chunk_id")
                ))
        except Exception:
            pass
