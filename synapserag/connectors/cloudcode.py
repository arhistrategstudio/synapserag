"""
Google Cloud Code & IDE Integration Bridge.
Exposes a lightweight programmatic and JSON-RPC dispatch protocol for IDE plugins and Cloud Code assistants.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..engine import SynapseEngine


class CloudCodeConnector:
    """
    Bridge for IDE extensions, Google Cloud Code, and code editor sidecars.
    Provides symbol-aware and file-context-aware retrieval.
    """
    def __init__(self, engine: "SynapseEngine"):
        self.engine = engine

    def handle_ide_request(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardized endpoint for IDE assistants:
        {
            "action": "query" | "ingest_active_file" | "find_symbol_definition",
            "params": {...}
        }
        """
        action = request_payload.get("action")
        params = request_payload.get("params", {})

        if action == "query":
            q = params.get("query", "")
            top_k = params.get("top_k", 5)
            mode = params.get("mode", "auto")
            res = self.engine.query(query=q, mode=mode, top_k=top_k)
            return {
                "status": "success",
                "matches": [
                    {
                        "uri": m.chunk.metadata.get("uri"),
                        "start_line": m.chunk.start_line,
                        "end_line": m.chunk.end_line,
                        "snippet": m.chunk.content,
                        "score": round(m.score, 4)
                    }
                    for m in res.matches
                ],
                "citations": [
                    {"uri": c.uri, "lines": f"{c.start_line}-{c.end_line}", "quote": c.exact_quote}
                    for c in res.citations
                ]
            }

        elif action == "ingest_active_file":
            content = params.get("content", "")
            uri = params.get("uri", "")
            doc = self.engine.ingest_text(text=content, uri=uri, title=params.get("title"))
            return {"status": "success", "doc_id": doc.doc_id, "uri": doc.uri}

        elif action == "find_symbol_definition":
            symbol = params.get("symbol", "")
            nodes = self.engine.graph_store.find_nodes_by_name([symbol])
            chunks = []
            for nid in nodes:
                node = self.engine.graph_store.nodes.get(nid)
                if node:
                    for cid in node.associated_chunks:
                        c = self.engine.vector_store.get_chunk(cid)
                        if c:
                            chunks.append({
                                "uri": c.metadata.get("uri"),
                                "start_line": c.start_line,
                                "end_line": c.end_line,
                                "content": c.content
                            })
            return {"status": "success", "symbol": symbol, "references": chunks}

        return {"status": "error", "message": f"Unsupported action: {action}"}
