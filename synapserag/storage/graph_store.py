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
        
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self.name_to_node[node.name.lower().strip()] = node.node_id
        for cid in node.associated_chunks:
            self.chunk_to_nodes[cid].add(node.node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        self.adjacency[edge.source_id].append(edge)
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
        clean_name = entity_name.lower().strip()
        if clean_name in self.name_to_node:
            nid = self.name_to_node[clean_name]
            node = self.nodes[nid]
            if chunk_id not in node.associated_chunks:
                node.associated_chunks.append(chunk_id)
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
            clean = kw.lower().strip()
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
        """Save graph structure to disk."""
        if not self.storage_dir:
            return
        data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "associated_chunks": n.associated_chunks,
                    "metadata": n.metadata
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "weight": e.weight,
                    "source_chunk_id": e.source_chunk_id
                }
                for edge_list in self.adjacency.values()
                for e in edge_list
                if not e.relation.startswith("rev_")
            ]
        }
        with open(self.storage_dir / "graph_store.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load graph from disk."""
        if not self.storage_dir:
            return
        path = self.storage_dir / "graph_store.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
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
