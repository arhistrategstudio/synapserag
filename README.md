# 🧠 SynapseRAG — Next-Gen Embeddable Tri-Brain RAG Engine

[![Tests](https://github.com/arhistrategstudio/synapserag/actions/workflows/tests.yml/badge.svg)](https://github.com/arhistrategstudio/synapserag/actions/workflows/tests.yml)
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

## 📥 Installation

The core engine has **zero external runtime dependencies** — it's built entirely on
the Python standard library. Optional extras add a real neural embedding backend
and the native MCP server.

```bash
git clone https://github.com/arhistrategstudio/synapserag.git
cd synapserag

# Core only (hash-based embeddings, zero dependencies)
pip install -e .

# With real embeddings (sentence-transformers/all-MiniLM-L6-v2 + ColBERT-style
# per-token vectors)
pip install -e ".[sentence-transformers,torch]"

# With the native MCP server
pip install -e ".[mcp]"

# Everything (embeddings + MCP + test tooling)
pip install -e ".[all,dev]"
```

Not yet published on PyPI — install from a local clone as shown above.

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

By default, `SynapseConfig.embedding_backend="auto"` uses the real
`sentence-transformers/all-MiniLM-L6-v2` model when the `sentence-transformers`
and `torch` extras are installed, and transparently falls back to a
zero-dependency hash-based embedder otherwise — the engine always works, with
or without the extras. Set `embedding_backend="hash"` to force the fast
dependency-free path (used by most of the test suite), or
`embedding_backend="sentence-transformers"` to require the real model and fail
loudly if it isn't available.

All storage (vector, graph, sparse) is persisted to `storage_dir` on disk by
default (`persist_on_write=True`); a fresh `SynapseEngine` pointed at the same
`storage_dir` reloads all three indexes without re-ingesting. Set
`persist_on_write=False` to keep everything in memory until you explicitly
call `engine.persist()`.

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

## ✅ Testing & Status

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

15/15 tests currently pass, covering the storage engines, the full ingest →
retrieve → verify pipeline, disk persistence/reload, and per-connector
edge cases (malformed tool calls, empty corpora, unknown tool/action names).
CI (GitHub Actions, `.github/workflows/tests.yml`) runs the suite on every
push/PR to `main` across Python 3.10–3.13, installing the `dev` and `mcp`
extras (the MCP connector test needs the `mcp` package) but leaving out
`sentence-transformers`/`torch`, so it also exercises the zero-dependency
hash-embedding fallback path.

**Known limitation:** on a very small corpus (e.g. a single ingested
document), the RRF-normalized confidence score used by the hallucination
circuit breaker can register as high-confidence even for a semantically
unrelated query, since it measures relative rank-consensus across the three
retrieval channels rather than absolute semantic similarity. This is more
reliable on larger, realistic corpora; improving small-corpus robustness is
tracked as future work.

---

## 📄 License
MIT License. Created by senior AI engineering team.
