# 🧠 SynapseRAG — Next-Gen Embeddable Tri-Brain RAG Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Native-green.svg)](https://modelcontextprotocol.io/)

**SynapseRAG** is an ultra-innovative, embeddable, zero-external-dependency RAG (Retrieval-Augmented Generation) micro-engine designed for software engineers, ML engineers, startup founders, and full-stack developers.

Unlike mainstream RAG frameworks (LangChain, LlamaIndex, isolated vector databases) which suffer from chunk context fragmentation, poor multi-hop reasoning, and heavy infrastructure overhead, **SynapseRAG** runs **inside** host applications, development environments, and autonomous agents.

---

## 🚀 Key Innovations

1. **Tri-Brain Unified Retrieval**:
   - **Brain A (Dense & Late-Interaction)**: ColBERTv2 / PLAID token-level MaxSim matching and contextual embeddings.
   - **Brain B (Neuro-Associative Graph)**: Vector-biased Personalized PageRank (PPR) for instant multi-hop associations without slow iterative LLM chains.
   - **Brain C (Sparse Inverted Index)**: BM25 / SPLADE lexical recovery for exact identifier, function, and code token search.
   - **Dynamic Reciprocal Rank Fusion (RRF)**: Adaptive score merging tuned to query intent.

2. **Hierarchical Contextual Late-Chunking**:
   - Long-context global attention preservation combined with dynamic semantic boundaries and situational context prepending.

3. **System 1 Memo-Clue Generator**:
   - Inspired by human associative memory (MemoRAG), generating speculative clues for vague, high-level, and abstract queries before retrieval.

4. **Deterministic Citation & Hallucination Circuit-Breaker**:
   - Token and character-level source span mapping (`file:line:char`) with confidence thresholds and fallback triggers.

5. **Universal In-App Connectors**:
   - **Native Model Context Protocol (MCP)** for Cursor, Claude Desktop, and Windsurf.
   - **OpenAI Agents Connector** (Function calling / Assistants / Swarm).
   - **Gemini Agents Connector** (Google GenAI / Vertex AI).
   - **DeepSeek Agents Connector** (DeepSeek-V3 / R1 reasoning tool format).
   - **Ollama Agents Connector** (Local LLMs).
   - **Google Cloud Code & IDE Integrations**.
   - **In-Process Python SDK** (Zero-network embedded mode).

---

## 📦 Architecture Overview

```
                           +-------------------------------------------------------------+
                           |                    HOST APPLICATION                         |
                           |   (Desktop App, IDE, Agent, Web Backend, CLI, Microservice) |
                           +-------------------------------------------------------------+
                                       |                      |                     |
                   [In-Process FFI / SDK]          [Model Context Protocol]    [Local IPC / gRPC]
                                       |                      |                     |
+---------------------------------------------------------------------------------------------------+
|                                      SYNAPSE RAG ENGINE CORE                                      |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. INGESTION & CONTEXT PIPELINE                                                                  |
|     + Contextual Boundary Detector (Semantic boundary recognition)                                |
|     + Late-Chunking Transformer Layer (Long-context global embeddings)                            |
|     + Entity & Relation Extractor (Lightweight Graph construction: Entity <-> Predicate <-> Chunk)|
|                                                                                                   |
|  2. THE TRI-BRAIN STORAGE & RETRIEVAL ENGINE                                                      |
|     + [Brain A] Dense & Late-Interaction Tensor Store (ColBERTv2 / PLAID fast MaxSim)            |
|     + [Brain B] Neuro-Associative Graph (Personalized PageRank & Spreading Activation)             |
|     + [Brain C] Inverted Sparse Index (BM25 / SPLADE lexical recovery)                            |
|                                                                                                   |
|  3. QUERY INTELLIGENCE & REASONING (System 1 + System 2)                                          |
|     + Memo-Clue Predictor: Generates latent clues for abstract queries                            |
|     + Multi-hop Associative Walker: Graph traversal without costly LLM hops                       |
|     + Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranker                                       |
|                                                                                                   |
|  4. VERIFICATION & CITATION CIRCUIT BREAKER                                                       |
|     + Grounding & Hallucination Scorer: Exact fact coverage analysis                              |
|     + Exact Span Citation Mapper: File, line, character tracking                                  |
|     + Adaptive Fallback Controller: Triggers active recovery if confidence is below threshold     |
+---------------------------------------------------------------------------------------------------+
```

---

## 🛠️ Quick Start

```python
from synapserag import SynapseEngine, SynapseConfig

# Initialize embedded engine (zero external servers required)
engine = SynapseEngine(config=SynapseConfig(storage_dir="./data/synapse_db"))

# Ingest documents or codebases with contextual late-chunking
engine.ingest(file_path="src/main.py")
engine.ingest(text="Project documentation...", doc_id="doc_01")

# Query with Tri-Brain fusion
results = engine.query(
    "How does the memory synchronization work across multi-hop nodes?",
    mode="hybrid", # "hybrid" | "late_interaction" | "graph_hop" | "lexical"
    top_k=10
)

for r in results.matches:
    print(f"[{r.score:.3f}] {r.file_path}:{r.line_number} -> {r.text_snippet}")
```

---

## 🔌 Connecting to AI Agents

### Cursor / Claude Desktop / Windsurf (via MCP)
Run the native MCP server:
```bash
python -m synapserag.mcp
```
Add to your `claude_desktop_config.json` or Cursor MCP settings:
```json
{
  "mcpServers": {
    "synapse-rag": {
      "command": "python",
      "args": ["-m", "synapserag.mcp", "--storage-dir", "./data/synapse_db"]
    }
  }
}
```

### OpenAI, Gemini, DeepSeek, Ollama & CloudCode
SynapseRAG exports native tool schemas ready for immediate injection:
```python
from synapserag.connectors import (
    OpenAIAgentConnector,
    GeminiAgentConnector,
    DeepSeekAgentConnector,
    OllamaAgentConnector
)

# Seamlessly bind to your favorite LLM agent framework
openai_tools = OpenAIAgentConnector(engine).get_tools()
gemini_tools = GeminiAgentConnector(engine).get_tools()
deepseek_tools = DeepSeekAgentConnector(engine).get_tools()
ollama_tools = OllamaAgentConnector(engine).get_tools()
```

---

## 📄 License
MIT License. Created by senior AI engineering team.
