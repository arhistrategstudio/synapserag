# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to adhere to [Semantic Versioning](https://semver.org/)
once it reaches a stable 1.0 API.

## [Unreleased]

## [0.1.0] - 2026-09-18

Initial release.

### Added
- Tri-Brain retrieval engine: dense/late-interaction vector store, neuro-associative
  graph store (vector-biased PPR), and sparse BM25 lexical index, fused via dynamic
  Reciprocal Rank Fusion.
- Hierarchical late-chunking ingestion pipeline with entity/relation extraction.
- System 1 memo-clue generator and adaptive query router.
- Deterministic citation/grounding engine (`file:line:char` mapping) and a
  hallucination circuit breaker, including a secondary absolute gate on raw dense
  cosine similarity (`min_semantic_similarity`) to guard against false rank-consensus
  on small corpora.
- Native MCP server plus OpenAI, Gemini, DeepSeek, Ollama, and Cloud Code agent
  connectors, and an in-process Python SDK.
- Zero-external-dependency core (`dependencies = []`); optional extras for
  `sentence-transformers`, `torch`, and `mcp`.
- Test suite (16 tests) and GitHub Actions CI across Python 3.10–3.13.
