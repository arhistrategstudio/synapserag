"""
DeepSeek Agents Connector (DeepSeek-V3 and DeepSeek-R1).
Formats tools and structures reasoning-focused outputs for DeepSeek agentic workflows.
"""

from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..engine import SynapseEngine


class DeepSeekAgentConnector:
    """
    Connects SynapseRAG to DeepSeek-V3 and DeepSeek-R1 models.
    Provides verified evidence blocks and reasoning clues that feed into DeepSeek's chain-of-thought.
    """
    def __init__(self, engine: "SynapseEngine"):
        self.engine = engine

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """Returns DeepSeek-compatible tool schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "synapse_knowledge_retrieval",
                    "description": "Deep multi-hop retrieval and code intelligence via SynapseRAG. Provides grounded evidence and associative links.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The exact question, code symbol, or conceptual relationship to look up."
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 5,
                                "description": "Number of evidence segments to retrieve."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def execute_tool(self, arguments: Dict[str, Any] | str) -> str:
        """Executes search and formats prompt-ready grounded evidence for DeepSeek reasoning."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        res = self.engine.query(
            query=arguments["query"],
            mode="auto",
            top_k=arguments.get("top_k", 5)
        )

        blocks = []
        if res.clues:
            blocks.append(f"### Latent Reasoning Clues:\n" + "\n".join(f"- {c}" for c in res.clues))

        blocks.append("### Grounded Evidence Passages:")
        for idx, m in enumerate(res.matches, 1):
            file_loc = f"{m.chunk.metadata.get('uri') or 'doc'}:{m.chunk.start_line}-{m.chunk.end_line}"
            blocks.append(
                f"[{idx}] Source: {file_loc} (Score: {m.score:.3f}, Channel: {m.channel})\n"
                f"{m.chunk.content}\n"
            )

        if res.circuit_breaker_triggered:
            blocks.append(f"WARNING: {res.warning}")

        return "\n".join(blocks)
