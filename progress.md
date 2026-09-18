# SynapseRAG — Dnevnik Razvoja i Napretka (Progress Tracker)

## 📌 Status Projekta
- **Datum pokretanja:** 2026-09-18
- **Trenutna faza:** Faza 8 završena (realni embedding, persistencija, connector testovi) — sledi Faza 9 (packaging/CI)
- **Status:** 🟢 Core implementiran i ojačan, testovi prolaze (15/15)

---

## 🗺️ Celokupan Plan Razvoja (Roadmap)

### Faza 1: Inicijalizacija i Osnovna Arhitektura
- [x] Detaljno istraživanje SOTA RAG tehnologija (HippoRAG, MemoRAG, LightRAG, ColBERT/PLAID, Late Chunking)
- [x] Definisanje arhitektonskog plana i inovativnog SynapseRAG koncepta
- [x] Inicijalizacija Git repozitorijuma i GitHub sinhronizacije
- [x] Kreiranje `progress.md` i osnovne strukture projekta
- [x] Implementacija konfiguracionog sistema (`synapserag/config.py`)
- [x] Implementacija ugrađenih tipova podataka, modela i šema (`synapserag/types.py`)

### Faza 2: Skladišno Jezgro — Tri-Brain Storage Engine
- [x] **Dense & Tensor Store (`synapserag/storage/vector_store.py`)**:
  - Podrška za guste vektore i ColBERT late-interaction tokene (MaxSim scoring).
  - Ugrađeno, samostalno skladište na disku (zero-external DB dependency).
- [x] **Neuro-Associative Graph (`synapserag/storage/graph_store.py`)**:
  - Inicijalizacija grafa relacija entiteta, predikata i izvornih tekstualnih segmenata.
  - Vektorski ponderisana matrica susedstva (Vector-Biased Adjacency Matrix).
- [x] **Sparse Lexical Index (`synapserag/storage/sparse_store.py`)**:
  - Brzi BM25 invertovani indeks za egzaktno poklapanje koda, ID-jeva i termina.

### Faza 3: Ingestion & Late-Chunking Pipeline
- [x] **Multi-Modal Embedder (`synapserag/ingest/embedder.py`)**:
  - Generisanje gustih vektora i ColBERT token embeddings.
- [x] **Hierarchical Late-Chunker (`synapserag/ingest/chunker.py`)**:
  - Očuvanje globalnog toka pažnje pre definisanja chunk granica.
  - Hijerarhijsko povezivanje roditelj-dete (Parent-Child chunking).
- [x] **Entity & Relation Extractor (`synapserag/ingest/graph_extractor.py`)**:
  - Ekstrakcija ključnih entiteta i veza u letu bez blokiranja ingestion procesa.

### Faza 4: Pretraživanje i Fuzija (Retrieval Core)
- [x] **Vector-Biased Personalized PageRank (`synapserag/retrieval/ppr.py`)**:
  - Algoritam matematičkog asocijativnog širenja kroz graf u jednom koraku.
- [x] **Late-Interaction MaxSim Matcher** (deo `storage/vector_store.py`):
  - Hirurško poređenje na nivou tokena.
- [x] **Dynamic Reciprocal Rank Fusion (`synapserag/retrieval/fusion.py`)**:
  - Inteligentno spajanje rezultata iz sva tri mozga (Dense + Graph + Sparse).
  - **Ispravljen bug (2026-09-18):** RRF skor je bio neispravno poređen sa apsolutnim pragom pouzdanosti u circuit breaker-u — RRF vrednosti su ograničene na `1/(k+1) ≈ 0.016`, pa je circuit breaker okidao na svaki upit. Dodata normalizacija fuzionisanog skora u opseg `[0, 1]`.
- [x] **Tri-Brain Retrieval Orchestrator (`synapserag/retrieval/engine.py`)**.

### Faza 5: Query Intelligence & Dual-System Clues
- [x] **System 1 Memo-Clue Generator (`synapserag/query/clue_engine.py`)**:
  - Formiranje spekulativnih hipoteza za nejasne i visokonivojske upite.
- [x] **Adaptive Query Router (`synapserag/query/router.py`)**:
  - Rutiranje: da li je upit egzaktan, asocijativan, globalan ili multi-hop.

### Faza 6: Ugradivi Konektori za Sve Agente (Universal Integration)
- [x] **Native MCP Server (`synapserag/connectors/mcp_server.py`, `synapserag/mcp.py`)**:
  - JSON-RPC stdio server za Cursor, Claude Desktop, Windsurf.
- [x] **OpenAI Agents Connector (`synapserag/connectors/openai_agent.py`)**:
  - Export alata i funkcija za OpenAI Assistants, Function Calling.
- [x] **Gemini Agents Connector (`synapserag/connectors/gemini_agent.py`)**:
  - Google GenAI Tool declarations za Gemini.
- [x] **DeepSeek Agents Connector (`synapserag/connectors/deepseek_agent.py`)**:
  - Optimizovan tool calling format za DeepSeek-V3/R1.
- [x] **Ollama Agents Connector (`synapserag/connectors/ollama_agent.py`)**:
  - Povezivanje sa lokalnim modelima koji se izvršavaju preko Ollama.
- [x] **CloudCode & IDE Connector (`synapserag/connectors/cloudcode.py`)**:
  - Integracija sa Google Cloud Code i razvojnim okruženjima.
- [x] **In-Process Python SDK (`synapserag/__init__.py`)**:
  - Direktan uvoz u bilo koji Python projekat u 1 liniji koda.

### Faza 7: Verifikacija, Determinističko Citiranje i Circuit Breaker
- [x] **Citation & Grounding Engine (`synapserag/verify/citations.py`)**:
  - Determinističko mapiranje izvora na nivou reda i karaktera (`file:line:char`).
- [x] **Hallucination Circuit Breaker (`synapserag/verify/circuit_breaker.py`)**:
  - Detekcija nepokrivenih tvrdnji i aktiviranje fallback mehanizma (sada radi nad normalizovanim skorom pouzdanosti).

### Faza 8: Testovi i Ojačavanje (Novo — sledeća faza)
- [x] Osnovni test paket (`tests/test_engine.py`, `tests/test_storage.py`) — 6/6 testova prolazi.
- [x] Prvi commit svih fajlova u Git i push na GitHub remote (`arhistrategstudio/synapserag`, commit `f515aa8`).
- [x] **Pravi embedding model (`synapserag/ingest/neural_backend.py`)**:
  - `SentenceTransformerBackend` — lenjo učitava `sentence-transformers/all-MiniLM-L6-v2` (384-dim, već keširan lokalno u
    `~/.cache/huggingface`), daje pooled dense embedding i per-token embedding (iz `last_hidden_state`) za ColBERT MaxSim.
  - Proces-wide model cache (`_MODEL_CACHE`) da se težine ne učitavaju ponovo za svaku instancu `SynapseEngine`-a.
  - `SynapseEngine._build_embedder()` automatski koristi ovaj backend kada je dostupan (`config.embedding_backend="auto"`,
    default), uz čist fallback na hash-based `MultiModalEmbedder` kad biblioteka/model nisu dostupni — zero-dependency
    garancija ostaje netaknuta. Nova config polja: `embedding_backend`, `embedding_model_name`.
  - Testovi (6/6) prolaze sa pravim embeddingom (~47s zbog učitavanja modela pri prvom pozivu, keširano nakon toga).
- [x] **Perzistencija/reload test (`tests/test_persistence.py`)**:
  - `test_engine_persist_and_reload_from_disk` — ingestuje dokument u jedan `SynapseEngine`, pravi potpuno nov engine
    iz istog `storage_dir` bez ponovnog ingest-a, i potvrđuje da su vector/graph/sparse indeksi i dalje tu i da upit radi.
  - `test_engine_no_persist_on_write_requires_explicit_persist` — potvrđuje ponašanje `persist_on_write=False`
    (podaci nisu na disku dok se eksplicitno ne pozove `engine.persist()`).
  - Testovi koriste `embedding_backend="hash"` da ostanu brzi i nezavisni od neuronskog modela.
- [x] **Testovi za pojedinačne konektore (`tests/test_connectors.py`)** — 7 novih edge-case testova:
  - OpenAI/Gemini konektori bacaju `ValueError` na nepoznat naziv alata; OpenAI konektor ispravno parsira JSON-string argumente.
  - DeepSeek konektor se ne ruši kad nema rezultata pretrage (prazan engine) — vraća string sa `WARNING` blokom umesto exception-a.
  - Ollama konektor ne puca na malformiran `tool_call` (nedostaje `function` ključ).
  - CloudCode konektor vraća `status: "error"` za nepoznatu akciju i prazne `references` za nepostojeći simbol, umesto da baci grešku.
  - **Poznato ograničenje (nije bag, zabeleženo za budući rad):** na malom korpusu (1 dokument) RRF-normalizovani confidence skor može biti visok i za semantički nepovezan upit, jer meri relativni rank-konsenzus kroz kanale a ne apsolutnu semantičku sličnost. Circuit breaker je pouzdaniji na većim, realnijim korpusima.
- Ukupno testova: **15/15 prolazi** (`tests/test_engine.py`, `tests/test_storage.py`, `tests/test_persistence.py`, `tests/test_connectors.py`).

---

## 📍 Gde smo stali (Current Milestone)
- Kompletna arhitektura implementirana kroz sve module: storage (vector/graph/sparse), ingest (chunker/embedder/graph_extractor),
  retrieval (PPR, fusion, tri-brain engine), query (clue engine, router), verify (citations, circuit breaker) i svih 6 konektora.
- Lokalni test paket (`pytest tests/`) prolazi 6/6.
- Otkriven i ispravljen bug: circuit breaker je poredio ne-normalizovani RRF fuzioni skor (max ~0.016) sa apsolutnim
  pragom pouzdanosti od 0.25, zbog čega je gotovo uvek okidao. Fuzioni skor je sada normalizovan u `[0, 1]` u `retrieval/fusion.py`.
- Kod još nije komitovan u Git (postoji lokalni repo + GitHub remote `origin` -> `arhistrategstudio/synapserag`, ali `synapserag/` i `tests/` su i dalje untracked).

---

## ⏭️ Šta je sledeće (Next Immediate Steps — Faza 9: Packaging & CI)
1. Dodati `pyproject.toml` (ili `setup.py`) da `synapserag` bude pip-instalabilan paket, sa jasno odvojenim core (zero-dep)
   i optional extras (`sentence-transformers`, `torch`, `mcp`) zavisnostima.
2. Dodati GitHub Actions workflow (`.github/workflows/tests.yml`) koji pokreće `pytest tests/` na svaki push/PR.
3. Razmotriti poboljšanje circuit breaker confidence metrike da bude robusnija na malim korpusima (trenutno RRF-normalizovan
   skor meri relativni rank-konsenzus kroz kanale, ne apsolutnu semantičku sličnost — vidi napomenu u Faza 8).
4. Pregledati i ažurirati `README.md` da odražava trenutno stanje implementacije (svi moduli + realni embedding backend).
