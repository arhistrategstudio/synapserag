"""
Google Gemini & Vertex AI Agents Connector.
Provides Gemini FunctionDeclaration tools and handlers for Gemini 1.5/2.0 agents.
"""

from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..engine import SynapseEngine


class GeminiAgentConnector:
    """
    Integrates SynapseRAG into Google GenAI / Gemini / Vertex AI agent workflows.
    """
    def __init__(self, engine: "SynapseEngine"):
        self.engine = engine

    def get_function_declarations(self) -> List[Dict[str, Any]]:
        """Returns Gemini function declarations for google.generativeai or vertexai SDK."""
        return [
            {
                "name": "synapse_rag_query",
                "description": "Execute precision retrieval across documents and code using SynapseRAG Tri-Brain search.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "The concept, question, or technical query to search for."
                        },
                        "mode": {
                            "type": "STRING",
                            "enum": ["hybrid", "late_interaction", "graph_hop", "lexical", "auto"],
                            "description": "Search strategy: 'auto', 'hybrid', 'late_interaction', 'graph_hop', 'lexical'."
                        },
                        "top_k": {
                            "type": "INTEGER",
                            "description": "Number of top matching chunks to retrieve."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "synapse_rag_ingest",
                "description": "Index new document or code content into SynapseRAG.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "text": {
                            "type": "STRING",
                            "description": "The raw text or code to be indexed."
                        },
                        "title": {
                            "type": "STRING",
                            "description": "Optional title for the document."
                        },
                        "uri": {
                            "type": "STRING",
                            "description": "Optional URI or file path."
                        }
                    },
                    "required": ["text"]
                }
            }
        ]

    def execute_call(self, function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handles execution of Gemini function calls."""
        if function_name == "synapse_rag_query":
            res = self.engine.query(
                query=args["query"],
                mode=args.get("mode", "auto"),
                top_k=int(args.get("top_k", 5))
            )
            return {
                "matches": [
                    {
                        "score": round(m.score, 4),
                        "channel": m.channel,
                        "file": m.chunk.metadata.get("uri"),
                        "lines": f"{m.chunk.start_line}-{m.chunk.end_line}",
                        "content": m.chunk.content
                    }
                    for m in res.matches
                ],
                "citations": [
                    {"quote": c.exact_quote, "line_range": f"{c.start_line}-{c.end_line}"}
                    for c in res.citations
                ],
                "circuit_breaker": res.circuit_breaker_triggered
            }
        elif function_name == "synapse_rag_ingest":
            doc = self.engine.ingest_text(
                text=args["text"],
                title=args.get("title"),
                uri=args.get("uri")
            )
            return {"status": "indexed", "doc_id": doc.doc_id}
        raise ValueError(f"Unknown Gemini tool: {function_name}")
