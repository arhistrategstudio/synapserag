import tempfile
from synapserag import SynapseEngine, SynapseConfig, RetrievalMode
from synapserag.connectors import (
    OpenAIAgentConnector,
    GeminiAgentConnector,
    DeepSeekAgentConnector,
    OllamaAgentConnector,
    CloudCodeConnector,
    create_mcp_server,
)


def test_engine_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SynapseConfig(storage_dir=tmpdir)
        engine = SynapseEngine(config=config)

        doc1_text = """
        SynapseRAG Architecture Overview:
        The Tri-Brain engine coordinates Dense Late-Interaction, Neuro-Associative Graphs, and BM25 Sparse search.
        Personalized PageRank propagates weights through entities like Transformer and Attention.
        ColBERT MaxSim ensures surgical token-level precision.
        """
        doc = engine.ingest_text(text=doc1_text, uri="docs/overview.md", title="Architecture Overview")
        assert doc.doc_id is not None

        # Query in hybrid mode
        result = engine.query("How does Personalized PageRank interact with Transformer?", mode="hybrid", top_k=3)
        assert len(result.matches) > 0
        assert len(result.citations) > 0
        assert result.citations[0].uri == "docs/overview.md"
        assert not result.circuit_breaker_triggered

        # Query with exact code keyword (auto routes to lexical or hybrid)
        code_result = engine.query("ColBERT MaxSim", mode="auto", top_k=2)
        assert len(code_result.matches) > 0


def test_agent_connectors():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SynapseConfig(storage_dir=tmpdir)
        engine = SynapseEngine(config=config)

        engine.ingest_text(
            text="The circuit breaker activates when retrieval confidence falls below threshold.",
            uri="docs/safety.md",
            title="Safety Specs"
        )

        # 1. OpenAI Connector
        openai_conn = OpenAIAgentConnector(engine)
        defs = openai_conn.get_tool_definitions()
        assert len(defs) == 2
        res_openai = openai_conn.execute_tool("synapse_rag_query", {"query": "circuit breaker"})
        assert len(res_openai["matches"]) > 0

        # 2. Gemini Connector
        gemini_conn = GeminiAgentConnector(engine)
        g_defs = gemini_conn.get_function_declarations()
        assert len(g_defs) == 2
        res_gemini = gemini_conn.execute_call("synapse_rag_query", {"query": "circuit breaker"})
        assert len(res_gemini["matches"]) > 0

        # 3. DeepSeek Connector
        deepseek_conn = DeepSeekAgentConnector(engine)
        ds_schema = deepseek_conn.get_tool_schema()
        assert len(ds_schema) == 1
        res_ds = deepseek_conn.execute_tool({"query": "circuit breaker"})
        assert "Grounded Evidence Passages" in res_ds

        # 4. Ollama Connector
        ollama_conn = OllamaAgentConnector(engine)
        o_tools = ollama_conn.get_tools()
        assert len(o_tools) == 1
        res_o = ollama_conn.handle_call({"function": {"name": "search", "arguments": {"query": "safety specs"}}})
        assert "content" in res_o

        # 5. CloudCode Connector
        cloud_conn = CloudCodeConnector(engine)
        res_cloud = cloud_conn.handle_ide_request({
            "action": "query",
            "params": {"query": "circuit breaker"}
        })
        assert res_cloud["status"] == "success"

        # 6. Native FastMCP Server
        mcp_server = create_mcp_server(engine)
        assert mcp_server is not None
