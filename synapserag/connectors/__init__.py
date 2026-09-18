"""
SynapseRAG Connectors Package — Universal In-App Agent Connectors.
Provides ready-to-use bindings for:
- Native Model Context Protocol (MCP) for Cursor, Claude Desktop, Windsurf
- OpenAI Agents & Assistants API
- Google Gemini & Vertex AI Agents
- DeepSeek-V3 & DeepSeek-R1 Agents
- Ollama Local LLMs
- Google Cloud Code & IDE Integrations
"""

from .mcp_server import create_mcp_server
from .openai_agent import OpenAIAgentConnector
from .gemini_agent import GeminiAgentConnector
from .deepseek_agent import DeepSeekAgentConnector
from .ollama_agent import OllamaAgentConnector
from .cloudcode import CloudCodeConnector

__all__ = [
    "create_mcp_server",
    "OpenAIAgentConnector",
    "GeminiAgentConnector",
    "DeepSeekAgentConnector",
    "OllamaAgentConnector",
    "CloudCodeConnector",
]
