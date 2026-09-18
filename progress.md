# SynapseRAG — Dnevnik Razvoja i Napretka (Progress Tracker)

## 📌 Status Projekta
- **Datum pokretanja:** 2026-09-18
- **Trenutna faza:** Faza 10 u toku (ojačavanje circuit breaker-a na malim korpusima)
- **Status:** 🟢 Core implementiran i ojačan, testovi prolaze (16/16), CI zeleno na GitHub Actions (Python 3.10-3.13)

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

### Faza 9: Packaging & CI (Završeno 2026-09-18)
- [x] **`pyproject.toml`**: `synapserag` je sada pip-instalabilan paket (`pip install -e .`).
  - Core paket ostaje **zero-dependency** (`dependencies = []`) — sve u `synapserag/` je stdlib-only, potvrđeno grep-om kroz sve module.
  - Optional extras: `sentence-transformers`, `torch`, `mcp` (pojedinačno ili `all`), plus `dev` (pytest).
  - **Bitna napomena:** `mcp` extra je pinovan na `>=1.0.0,<2.0.0` — `mcp` 2.x je preimenovao `FastMCP` u `MCPServer`
    (`mcp.server.fastmcp` više ne postoji), a naš `connectors/mcp_server.py` je pisan protiv 1.x `FastMCP` API-ja.
    Bez pina, `pip install ".[mcp]"` povlači 2.x i `test_agent_connectors` tiho puca (ImportError se guta u
    `try/except` pa `create_mcp_server` baca grešku pri pozivu). Otkriveno i popravljeno tokom CI podešavanja.
- [x] **GitHub Actions (`.github/workflows/tests.yml`)**: pokreće `pytest tests/` na push/PR prema `main`, matrica Python
  3.10–3.13. Instalira `.[dev,mcp]` (bez `sentence-transformers`/`torch`) — ovo namerno tera testove da koriste
  hash-based embedding fallback, pa CI usput validira i zero-dependency putanju. Zeleno na sve 4 verzije Pythona.
- [x] **`README.md`** ažuriran: CI badge, sekcija za instalaciju (`pip install -e ".[extras]"`), objašnjenje
  `embedding_backend="auto"` fallback ponašanja, napomena o persistenciji (`persist_on_write`), sekcija
  Testing & Status sa poznatim ograničenjem circuit breaker-a na malim korpusima.
- Sav kod je već bio komitovan i pushovan pre ove faze (prethodna napomena o untracked fajlovima u ovom progress.md je bila zastarela — `git status` na početku Faze 9 je bio clean).

### Faza 10: Ojačavanje Circuit Breaker-a na Malim Korpusima (U toku, 2026-09-18)
- [x] **Sekundarni, apsolutni gejt na sirovu dense cosine sličnost (`synapserag/verify/circuit_breaker.py`)**:
  - Rešava poznato ograničenje iz Faze 8/9: RRF-normalizovani fused skor meri relativni rank-konsenzus kroz tri
    kanala, ne apsolutnu semantičku sličnost. Na malom korpusu (npr. jedan dokument) nepovezan upit i dalje trivijalno
    rangira jedini postojeći chunk na #1 mesto u dense kanalu (dok sparse/graph ne nađu ništa), pa je fused skor mogao
    proći prag pouzdanosti iako match nije stvarno relevantan.
  - `HallucinationCircuitBreaker.evaluate()` sada, pored postojeće provere `min_confidence_threshold` na fused skoru,
    dodatno proverava sirov `dense_score` (cosine sličnost) top match-a protiv novog `min_semantic_similarity` praga
    — ali samo kad je `channel == "hybrid"` i `dense_score != 0.0` (tj. dense kanal je stvarno učestvovao).
  - Novo config polje: `SynapseConfig.min_semantic_similarity` (default `0.22`), povezano kroz `SynapseEngine.__init__`.
  - **Kalibracija (izmereno lokalno, ne pretpostavljeno):** za `all-MiniLM-L6-v2` nepovezani parovi rečenica daju
    dense_score ~0.15-0.16, dok stvarno relevantni upiti daju ~0.72-0.75 — jasna margina oko praga 0.22. Za
    zero-dependency hash embedder (fallback), nepovezani tekstovi daju ~0.37 (viši šum jer je to prosek nezavisnih
    hash-projektovanih vektora kratkih tekstova bez naučene semantike), dok relevantni upiti i dalje daju ~0.82 —
    gejt i dalje radi, ali sa manjom marginom.
  - Novi regresioni test (`tests/test_connectors.py::test_circuit_breaker_catches_small_corpus_false_consensus`):
    jedan dokument o pekarskom inventaru, upit o termodinamici crnih rupa (nula preklapanja u temi) sada ispravno
    aktivira circuit breaker; kontrolni upit o zalihama brašna/šećera na istom korpusu ispravno NE aktivira breaker.
    Test koristi `pytest.importorskip("sentence_transformers")` i preskače se u CI (koji namerno ne instalira
    `sentence-transformers`/`torch` da bi testirao hash fallback putanju) — pokreće se lokalno ili sa
    `pip install -e ".[sentence-transformers]"`.
  - Ukupno testova: **16/16 prolazi lokalno** (15 postojećih + 1 novi; CI i dalje vidi 15/16 jer 1 test preskače).
  - README.md ažuriran (sekcija Testing & Status) sa objašnjenjem novog gejta i njegove kalibracije po backend-u.
- [x] **Infrastruktura za PyPI objavljivanje (2026-09-18)**:
  - `LICENSE` (MIT, nedostajao je iako je `pyproject.toml` deklarisao `license = "MIT"`) — potrebno za PyPI/quality checks.
  - Verifikovano lokalno: `python -m build` pravi ispravan sdist + wheel (`synapserag-0.1.0`), `twine check dist/*` → PASSED za oba.
  - **`.github/workflows/publish.yml`**: novi GitHub Actions workflow, okida se na push taga `v*`, build-uje distribuciju i
    objavljuje na PyPI preko [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, `id-token: write`) —
    bez čuvanja dugotrajnog API tokena kao GitHub secret-a.
  - `CHANGELOG.md` dodat (Keep a Changelog format), sa `0.1.0` unosom koji sumira sve što je do sada implementirano.
  - README ažuriran: nova sekcija "Releasing (maintainers)" sa tačnim koracima za cutting a release.
- [x] **Prvi PyPI release objavljen (2026-09-18)**:
  - Vlasnik je uključio 2FA na PyPI nalogu (preduslov koji PyPI zahteva za Trusted Publisher podešavanje).
  - Pending trusted publisher registrovan na pypi.org (`arhistrategstudio/synapserag`, workflow `publish.yml`,
    environment `pypi`) preko browser automation (Claude in Chrome) — agent je popunio i submitovao formu.
  - Tag `v0.1.0` push-ovan → `.github/workflows/publish.yml` se okinuo, build + publish job-ovi su prošli (zeleno),
    paket je potvrđen na PyPI JSON API-ju (`synapserag 0.1.0`, `synapserag-0.1.0-py3-none-any.whl`).
  - `pip install synapserag` sada radi bez kloniranja repoa. README ažuriran (install sekcija koristi `pip install
    synapserag` umesto `git clone` + `pip install -e .`, dodat PyPI verzija badge).
  - **Napomena o bezbednosti:** tokom 2FA setup-a, PyPI recovery-codes fajl je preuzet direktno u root ovog git
    repoa (`PyPI-Recovery-Codes-*.txt`). Agent ga NIJE komitovao/pushovao, i upozorio je korisnika da ga premesti
    izvan repoa (npr. u password manager) čim pre — osetljiv fajl, ne pripada verzionisanom kodu. Korisnik je
    potvrdio da su kodovi bezbedno sačuvani; fajl je obrisan iz repo foldera.
- [x] **Otkriven i ispravljen bug u README Quick Start primeru (2026-09-18)**:
  - Primer u README-u je pozivao `engine.ingest(file_path=..., text=..., doc_id=...)` i čitao `r.file_path` /
    `r.line_number` / `r.text_snippet` sa `RetrievalMatch` objekata — nijedno od toga ne postoji u stvarnom API-ju.
    Prava metoda su `engine.ingest_file(path)` / `engine.ingest_text(text, uri=...)`, a file/line/citat podaci se
    nalaze na paralelnoj `results.citations` listi (`CitationSpan.uri`, `.start_line`, `.exact_quote`), ne na
    `results.matches` (koji nose samo `score`/`dense_score`/`graph_score`/`sparse_score`/`chunk`).
  - Otkriveno pri pravljenju konkretnog primera (`examples/quickstart_demo.py`) koji SynapseRAG koristi na
    sopstvenom izvornom kodu (`circuit_breaker.py`, `fusion.py`, `README.md`) — README kod se nikad ranije nije
    stvarno pokretao/testirao, samo je pisan uz implementaciju.
  - README Quick Start sekcija ispravljena da odgovara stvarnom API-ju i sada linkuje na `examples/quickstart_demo.py`.
  - Demo lokalno pokrenut i verifikovan: ingestuje 3 fajla iz repoa, postavlja 3 pitanja, vraća relevantne odgovore
    sa tačnim `file:line` citatima (npr. pitanje o circuit breaker-u ispravno pogađa `circuit_breaker.py:35` i `:46`).

---

## 📍 Gde smo stali (Current Milestone)
- Kompletna arhitektura implementirana kroz sve module: storage (vector/graph/sparse), ingest (chunker/embedder/graph_extractor),
  retrieval (PPR, fusion, tri-brain engine), query (clue engine, router), verify (citations, circuit breaker) i svih 6 konektora.
- Lokalni test paket (`pytest tests/`) prolazi 16/16 (CI vidi 15/16 — 1 test se namerno preskače bez `sentence-transformers`),
  i CI (GitHub Actions) je zeleno na Python 3.10-3.13.
- Projekat je pip-instalabilan (`pyproject.toml`) sa zero-dep core i optional extras.
- Otkriven i ispravljen bug (Faza 4): circuit breaker je poredio ne-normalizovani RRF fuzioni skor (max ~0.016) sa apsolutnim
  pragom pouzdanosti od 0.25, zbog čega je gotovo uvek okidao. Fuzioni skor je sada normalizovan u `[0, 1]` u `retrieval/fusion.py`.
- Otkriven i ispravljen bug (Faza 9): `mcp` extra bez gornje granice verzije povlači mcp 2.x koji je preimenovao
  `FastMCP` → `MCPServer`, čime bi `create_mcp_server()` tiho prestao da radi. Pinovano na `mcp>=1.0.0,<2.0.0`.
- Faza 10 (u toku): circuit breaker sada ima sekundarni gejt na sirovu dense cosine sličnost (`min_semantic_similarity`,
  default 0.22) da uhvati slučaj lažnog rank-konsenzusa na malom korpusu — vidi detalje iznad.
- Sve promene su komitovane i pushovane na `origin/main` (`arhistrategstudio/synapserag`) nakon svakog završenog koraka.

---

## ⏭️ Šta je sledeće (Next Immediate Steps — Faza 10 ideje)
1. ~~Objaviti na PyPI~~ — **urađeno** (`pip install synapserag` radi, verzija 0.1.0 live na PyPI-u od 2026-09-18).
   Vidi detalje u Fazi 10 iznad.
2. ~~Poboljšati circuit breaker confidence metriku da bude robusnija na malim korpusima~~ — **urađeno** (sekundarni
   apsolutni gejt na dense cosine sličnost, vidi Fazu 10 iznad). Moguć budući rad: kalibrisati/dokumentovati ponašanje
   i za `graph_score`/`sparse_score` kanale (trenutno gejt koristi samo `dense_score`, jer je jedini kanal sa
   dobro definisanom apsolutnom skalom u [0,1]; BM25 je neograničen, a PPR aktivaciona masa zavisi od veličine grafa).
3. **Pratiti `mcp` SDK 2.x migraciju** — trenutno pinovano na `<2.0.0` da radi sa `FastMCP`; kad/ako se odluči migracija
   na `MCPServer` API iz 2.x, treba ažurirati `synapserag/connectors/mcp_server.py` i onda skinuti pin u `pyproject.toml`.
4. ~~Razmotriti dodavanje `CHANGELOG.md` i verzionisanje releasa~~ — **urađeno** (`CHANGELOG.md` dodat, Keep a Changelog
   format, `0.1.0` unos; verzionisanje releasa je sada vezano za `pyproject.toml` verziju + git tag preko `publish.yml`).
