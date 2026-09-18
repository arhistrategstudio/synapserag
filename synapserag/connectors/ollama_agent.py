"""
Ollama Local LLM Connector.
Provides function calling tools formatted for Ollama Python SDK and local REST API.
"""

from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..engine import SynapseEngine


class OllamaAgentConnector:
    """
    Connects SynapseRAG to local models running on Ollama (Llama 3.2, Qwen 2.5, Mistral, etc.).
    Operates 100% locally and offline.
    """
    def __init__(self, engine: "SynapseEngine"):
        self.engine = engine

    def get_tools(self) -> List[Dict[str, Any]]:
        """Returns tools list for `ollama.chat(..., tools=...)`."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Searches the local SynapseRAG database for documentation, code, and factual records.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search terms or question."
                            },
                            "limit": {
                                "type": "integer",
                                "default": 4,
                                "description": "Number of results."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def handle_call(self, tool_call: Dict[str, Any]) -> str:
        """Processes tool call object returned by Ollama model."""
        fn = tool_call.get("function", {})
        args = fn.get("arguments", {})
        query = args.get("query", "")
        limit = args.get("limit", 4)

        res = self.engine.query(query=query, mode="auto", top_k=limit)
        results = [
            {
                "file": m.chunk.metadata.get("uri"),
                "lines": f"{m.chunk.start_line}-{m.chunk.end_line}",
                "content": m.chunk.content
            }
            for m in res.matches
        ]
        return json.dumps(results)
