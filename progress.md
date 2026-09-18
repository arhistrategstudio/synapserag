# SynapseRAG — Dnevnik Razvoja i Napretka (Progress Tracker)

## 📌 Status Projekta
- **Datum pokretanja:** 2026-09-18
- **Trenutna faza:** Faza 1 (Inicijalizacija arhitekture i skladišnog jezgra)
- **Status:** 🟡 U izradi (In Progress)

---

## 🗺️ Celokupan Plan Razvoja (Roadmap)

### Faza 1: Inicijalizacija i Osnovna Arhitektura
- [x] Detaljno istraživanje SOTA RAG tehnologija (HippoRAG, MemoRAG, LightRAG, ColBERT/PLAID, Late Chunking)
- [x] Definisanje arhitektonskog plana i inovativnog SynapseRAG koncepta
- [x] Inicijalizacija Git repozitorijuma i GitHub sinhronizacije
- [x] Kreiranje `progress.md` i osnovne strukture projekta
- [ ] Implementacija konfiguracionog sistema (`synapserag/config.py`)
- [ ] Implementacija ugrađenih tipova podataka, modela i šema (`synapserag/types.py`)

### Faza 2: Skladišno Jezgro — Tri-Brain Storage Engine
- [ ] **Dense & Tensor Store (`synapserag/storage/vector_store.py`)**:
  - Podrška za guste vektore i ColBERT late-interaction tokene (MaxSim scoring).
  - Ugrađeno, samostalno skladište na disku (zero-external DB dependency).
- [ ] **Neuro-Associative Graph (`synapserag/storage/graph_store.py`)**:
  - Inicijalizacija grafa relacija entiteta, predikata i izvornih tekstualnih segmenata.
  - Vektorski ponderisana matrica susedstva (Vector-Biased Adjacency Matrix).
- [ ] **Sparse Lexical Index (`synapserag/storage/sparse_store.py`)**:
  - Brzi BM25 / SPLADE invertovani indeks za egzaktno poklapanje koda, ID-jeva i termina.

### Faza 3: Ingestion & Late-Chunking Pipeline
- [ ] **Document Readers & Context Encoders (`synapserag/ingest/parser.py`)**:
  - Učitavanje teksta, koda, Markdown-a, PDF-a.
  - Generisanje globalnog situacionog konteksta po fajlu (Contextual Retrieval).
- [ ] **Hierarchical Late-Chunker (`synapserag/ingest/chunker.py`)**:
  - Očuvanje globalnog toka pažnje pre definisanja chunk granica.
  - Hijerarhijsko povezivanje roditelj-dete (Parent-Child chunking).
- [ ] **Entity & Relation Extractor (`synapserag/ingest/graph_extractor.py`)**:
  - Ekstrakcija ključnih entiteta i veza u letu bez blokiranja ingestion procesa.

### Faza 4: Pretraživanje i Fuzija (Retrieval Core)
- [ ] **Vector-Biased Personalized PageRank (`synapserag/retrieval/ppr.py`)**:
  - Algoritam matematičkog asocijativnog širenja kroz graf u jednom koraku.
- [ ] **Late-Interaction MaxSim Matcher (`synapserag/retrieval/late_interaction.py`)**:
  - Hirurško poređenje na nivou tokena.
- [ ] **Dynamic Reciprocal Rank Fusion (`synapserag/retrieval/fusion.py`)**:
  - Inteligentno spajanje rezultata iz sva tri mozga (Dense + Graph + Sparse).

### Faza 5: Query Intelligence & Dual-System Clues
- [ ] **System 1 Memo-Clue Generator (`synapserag/query/clue_engine.py`)**:
  - Formiranje spekulativnih hipoteza za nejasne i visokonivojske upite.
- [ ] **Adaptive Query Router (`synapserag/query/router.py`)**:
  - Rutiranje: da li je upit egzaktan, asocijativan, globalan ili multi-hop.

### Faza 6: Ugradivi Konektori za Sve Agente (Universal Integration)
- [ ] **Native MCP Server (`synapserag/connectors/mcp_server.py`)**:
  - JSON-RPC stdio/SSE server za Cursor, Claude Desktop, Windsurf.
- [ ] **OpenAI Agents Connector (`synapserag/connectors/openai_agent.py`)**:
  - Export alata i funkcija za OpenAI Assistants, Function Calling i Swarm.
- [ ] **Gemini Agents Connector (`synapserag/connectors/gemini_agent.py`)**:
  - Google GenAI Tool declarations za Gemini 1.5/2.0 i Vertex AI.
- [ ] **DeepSeek Agents Connector (`synapserag/connectors/deepseek_agent.py`)**:
  - Optimizovan tool calling format za DeepSeek-V3 i DeepSeek-R1.
- [ ] **Ollama Agents Connector (`synapserag/connectors/ollama_agent.py`)**:
  - Povezivanje sa lokalnim modelima koji se izvršavaju preko Ollama.
- [ ] **CloudCode & IDE Connector (`synapserag/connectors/cloudcode.py`)**:
  - Integracija sa Google Cloud Code i razvojnim okruženjima.
- [ ] **In-Process Python SDK (`synapserag/__init__.py`)**:
  - Direktan uvoz u bilo koji Python projekat u 1 liniji koda.

### Faza 7: Verifikacija, Determinističko Citiranje i Circuit Breaker
- [ ] **Citation & Grounding Engine (`synapserag/verify/citations.py`)**:
  - Determinističko mapiranje izvora na nivou reda i karaktera (`file:line:char`).
- [ ] **Hallucination Circuit Breaker (`synapserag/verify/circuit_breaker.py`)**:
  - Detekcija nepokrivenih tvrdnji i aktiviranje fallback mehanizma.

---

## 📍 Gde smo stali (Current Milestone)
- Završena konceptualizacija, komparativno istraživanje i arhitektonski nacrt.
- Pokrenuto postavljanje repozitorijuma, GitHub sinhronizacija i pisanje koda Faze 1 i Faze 2.

---

## ⏭️ Šta je sledeće (Next Immediate Steps)
1. Inicijalizacija lokalnog Git repozitorijuma i kreiranje GitHub remote repozitorijuma preko `gh repo create`.
2. Kreiranje osnovne modularne strukture paketa `synapserag`.
3. Implementacija `synapserag/types.py` i `synapserag/config.py`.
4. Implementacija ugrađenog Tri-Brain skladišta (vektori, graf relacija, BM25 indeks) sa nula eksternih server zavisnosti.
