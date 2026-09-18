"""
Native Model Context Protocol (MCP) Server for SynapseRAG.
Enables instant plug-and-play integration with Cursor, Claude Desktop, Windsurf, and any MCP-compliant client.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import json

try:
    from mcp.server.fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False

if TYPE_CHECKING:
    from ..engine import SynapseEngine


def create_mcp_server(engine: "SynapseEngine", server_name: str = "synapse-rag") -> FastMCP:
    """
    Creates and configures an MCP server exposing SynapseRAG tools.
    """
    if not HAS_FASTMCP:
        raise ImportError("mcp package is required. Install via `pip install mcp`.")

    mcp = FastMCP(server_name)

    @mcp.tool()
    def rag_query(query: str, mode: str = "hybrid", top_k: int = 5) -> str:
        """
        Query the SynapseRAG knowledge base.
        
        Args:
            query: The user inquiry or code question.
            mode: Retrieval mode: 'hybrid' (Tri-Brain), 'late_interaction' (ColBERT MaxSim), 'graph_hop' (PPR), or 'lexical' (BM25).
            top_k: Number of relevant evidence chunks to return.
        """
        result = engine.query(query, mode=mode, top_k=top_k)
        output = {
            "query": result.query,
            "mode": result.retrieval_mode.value,
            "circuit_breaker_triggered": result.circuit_breaker_triggered,
            "clues": result.clues,
            "matches": [
                {
                    "score": m.score,
                    "channel": m.channel,
                    "file_uri": m.chunk.metadata.get("uri"),
                    "lines": f"{m.chunk.start_line}-{m.chunk.end_line}",
                    "content": m.chunk.content
                }
                for m in result.matches
            ],
            "citations": [
                {
                    "quote": c.exact_quote,
                    "file": c.uri,
                    "line_range": f"{c.start_line}:{c.end_line}",
                    "confidence": round(c.confidence_score, 3)
                }
                for c in result.citations
            ]
        }
        return json.dumps(output, indent=2)

    @mcp.tool()
    def rag_ingest_text(text: str, uri: str = "", title: str = "") -> str:
        """
        Ingest text or code directly into the SynapseRAG engine.
        
        Args:
            text: Raw document or code content.
            uri: Optional file path or URL.
            title: Optional title of the document.
        """
        doc = engine.ingest_text(text=text, uri=uri or None, title=title or None)
        return json.dumps({
            "status": "success",
            "doc_id": doc.doc_id,
            "title": doc.title,
            "uri": doc.uri
        })

    @mcp.tool()
    def rag_graph_traverse(entity_name: str) -> str:
        """
        Explore knowledge graph relations and multi-hop connections for a specific entity or symbol.
        
        Args:
            entity_name: Name of the class, function, concept or entity.
        """
        clean_name = entity_name.lower().strip()
        nodes = engine.graph_store.find_nodes_by_name([clean_name])
        if not nodes:
            return json.dumps({"status": "not_found", "message": f"Entity '{entity_name}' not in graph."})

        details = []
        for nid in nodes:
            node = engine.graph_store.nodes.get(nid)
            if not node:
                continue
            edges = engine.graph_store.adjacency.get(nid, [])
            connected = [
                {"target": engine.graph_store.nodes[e.target_id].name, "relation": e.relation, "weight": e.weight}
                for e in edges if e.target_id in engine.graph_store.nodes
            ]
            details.append({
                "entity": node.name,
                "type": node.node_type,
                "associated_chunks_count": len(node.associated_chunks),
                "connections": connected
            })

        return json.dumps({"entities": details}, indent=2)

    return mcp
