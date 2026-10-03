# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to adhere to [Semantic Versioning](https://semver.org/)
once it reaches a stable 1.0 API.

## [Unreleased]

## [0.2.0] - 2026-10-03

Fixes found while indexing a real Serbian legal corpus (laws, court decisions, textbooks;
~18 MB of text). **Existing indexes must be rebuilt** to benefit from the tokenizer fix;
old JSON indexes are still readable and are migrated to SQLite on the next `persist()`.

### Fixed
- **Non-ASCII text was silently dropped by every tokenizer.** The sparse (BM25), hash
  embedder, graph and clue tokenizers only matched `[a-zA-Z0-9_-]`, so Cyrillic produced
  zero tokens and words with diacritics were cut ("krađa" -> "kra", "a"). All channels now
  use `synapserag.text.normalize_text` / `word_tokens`: Unicode-aware, lowercase, Serbian
  Cyrillic transliterated to Latin, diacritics stripped — a Latin-script query now finds a
  Cyrillic document and vice versa.
- **Wrong citation line/char offsets.** Paragraphs longer than the chunk size were cut into
  word windows re-joined with single spaces, so `raw_text.find()` failed and every later
  chunk inherited a drifting cursor (e.g. a passage on line 1173 was cited as line 1).
  Chunks are now exact `(start_char, end_char)` spans of the original text; line numbers
  are resolved by binary search on `"\n"` only (form feeds from PDF extraction no longer
  shift line numbers). Also fixes a redundant trailing window.
- Graph entity extraction now recognizes capitalized words in any script.

### Changed
- **Storage moved from JSON to SQLite** (`<storage_dir>/synapse.db`, standard library
  only): float32 dense vectors and float16 per-token vectors stored as binary blobs, and
  `persist()` writes only rows added since the last persist instead of re-serializing the
  whole index. Measured on the same 3,228-chunk legal sample (hash backend, late
  interaction on): 826 MB -> 217 MB on disk, persist 42 s -> 1.2 s, reload 0.5 s.
- In memory, vectors are kept as compact `array`/`bytes` instead of Python float lists;
  `Chunk.dense_vector` / `Chunk.token_embeddings` are cleared after `add_chunk` — use
  `EmbeddedVectorStore.get_dense_vector()` / `get_token_embeddings()` to read them.
  The numpy dense matrix is cached between queries.
- **Default embedding model is now `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
  (384-dim, 50+ languages) instead of the English-only `all-MiniLM-L6-v2`. On Serbian
  text the English model scored a relevant passage (0.55) no higher than unrelated ones
  (0.43–0.53); the multilingual model separates them (0.61 vs 0.02–0.10).
- `min_semantic_similarity` now defaults to `None` = auto floor per setup (neural dense
  0.40, neural MaxSim 0.60, hash 0.22), provisionally calibrated on a small Serbian legal
  sample; an explicit float still overrides it.

### Notes
- Per-token (ColBERT) vectors remain the dominant storage cost. On a 3,974-chunk legal
  sample with the multilingual model: late interaction on = 335 MB, 9/10 expected passages
  in top-5; late interaction off (dense only) = 49 MB, 8/10. For large corpora consider
  `enable_late_chunking=False`.
- No stemming yet: BM25 matches exact (normalized) word forms, so inflected forms
  ("krađa" / "krađu") only meet through the dense channel.

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
