"""
OpenAI Agents & Function Calling Connector.
Provides official JSON schemas and tool execution handlers for OpenAI Assistants, Function Calling, and Swarm.
"""

from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..engine import SynapseEngine


class OpenAIAgentConnector:
    """
    Connects SynapseRAG directly into OpenAI Agents, GPT-4o, and Assistants API.
    """
    def __init__(self, engine: "SynapseEngine"):
        self.engine = engine

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns tool declarations formatted for the OpenAI API `tools` parameter."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "synapse_rag_query",
                    "description": "Perform precision Tri-Brain RAG search across the ingested knowledge base, code, and documentation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query, question, or technical concept to retrieve evidence for."
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["hybrid", "late_interaction", "graph_hop", "lexical", "auto"],
                                "default": "auto",
                                "description": "Retrieval mode: 'auto' (recommended), 'hybrid' (Tri-Brain), 'late_interaction' (ColBERT MaxSim), 'graph_hop' (PPR), or 'lexical' (BM25)."
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 5,
                                "description": "Maximum number of grounded evidence chunks to return."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "synapse_rag_ingest",
                    "description": "Ingest new raw text, code snippet, or file content into the RAG engine in real time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The content to index."
                            },
                            "title": {
                                "type": "string",
                                "description": "Optional title or summary identifier for the document."
                            },
                            "uri": {
                                "type": "string",
                                "description": "Optional file path or URI."
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any] | str) -> Dict[str, Any]:
        """Dispatches an OpenAI tool call and returns serializable payload."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        if name == "synapse_rag_query":
            res = self.engine.query(
                query=arguments["query"],
                mode=arguments.get("mode", "auto"),
                top_k=arguments.get("top_k", 5)
            )
            return {
                "query": res.query,
                "mode": res.retrieval_mode.value,
                "circuit_breaker_triggered": res.circuit_breaker_triggered,
                "matches": [
                    {
                        "score": round(m.score, 4),
                        "file": m.chunk.metadata.get("uri"),
                        "lines": f"{m.chunk.start_line}-{m.chunk.end_line}",
                        "content": m.chunk.content
                    }
                    for m in res.matches
                ],
                "citations": [
                    {
                        "quote": c.exact_quote,
                        "file": c.uri,
                        "line_range": f"{c.start_line}-{c.end_line}",
                        "confidence": round(c.confidence_score, 3)
                    }
                    for c in res.citations
                ]
            }
        elif name == "synapse_rag_ingest":
            doc = self.engine.ingest_text(
                text=arguments["text"],
                title=arguments.get("title"),
                uri=arguments.get("uri")
            )
            return {"status": "success", "doc_id": doc.doc_id, "title": doc.title}
        else:
            raise ValueError(f"Unknown tool: {name}")
