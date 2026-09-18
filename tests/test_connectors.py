import json
import tempfile

import pytest

from synapserag import SynapseEngine, SynapseConfig
from synapserag.connectors import (
    OpenAIAgentConnector,
    GeminiAgentConnector,
    DeepSeekAgentConnector,
    OllamaAgentConnector,
    CloudCodeConnector,
)


def _engine(tmpdir: str) -> SynapseEngine:
    # Hash backend keeps connector edge-case tests fast/independent of the neural model.
    return SynapseEngine(config=SynapseConfig(storage_dir=tmpdir, embedding_backend="hash"))


def test_openai_connector_unknown_tool_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = OpenAIAgentConnector(_engine(tmpdir))
        with pytest.raises(ValueError):
            conn.execute_tool("not_a_real_tool", {"query": "anything"})


def test_openai_connector_accepts_json_string_arguments():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = _engine(tmpdir)
        engine.ingest_text(text="OpenAI connector accepts JSON-encoded tool arguments.", uri="docs/openai.md")
        conn = OpenAIAgentConnector(engine)
        res = conn.execute_tool("synapse_rag_query", json.dumps({"query": "JSON-encoded arguments"}))
        assert "matches" in res


def test_gemini_connector_unknown_function_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = GeminiAgentConnector(_engine(tmpdir))
        with pytest.raises(ValueError):
            conn.execute_call("not_a_real_function", {"query": "anything"})


def test_deepseek_connector_handles_no_matches_gracefully():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Nothing ingested: retrieval must legitimately find zero matches and the
        # circuit breaker must trip, but the connector still has to return a
        # well-formed string rather than raising.
        conn = DeepSeekAgentConnector(_engine(tmpdir))
        output = conn.execute_tool({"query": "anything at all"})
        assert isinstance(output, str)
        assert "WARNING" in output
        assert "Grounded Evidence Passages" in output


def test_ollama_connector_handles_missing_function_payload_gracefully():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = _engine(tmpdir)
        engine.ingest_text(text="Ollama connector must not crash on malformed tool calls.", uri="docs/ollama.md")
        conn = OllamaAgentConnector(engine)

        # Malformed tool_call missing the "function" key entirely.
        res = conn.handle_call({})
        parsed = json.loads(res)
        assert isinstance(parsed, list)


def test_cloudcode_connector_unsupported_action_returns_error_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = CloudCodeConnector(_engine(tmpdir))
        res = conn.handle_ide_request({"action": "delete_everything", "params": {}})
        assert res["status"] == "error"
        assert "delete_everything" in res["message"]


def test_cloudcode_connector_find_symbol_definition_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = _engine(tmpdir)
        engine.ingest_text(text="A document without the symbol being searched for.", uri="docs/x.md")
        conn = CloudCodeConnector(engine)
        res = conn.handle_ide_request({
            "action": "find_symbol_definition",
            "params": {"symbol": "TotallyUnknownSymbolXYZ"}
        })
        assert res["status"] == "success"
        assert res["references"] == []
