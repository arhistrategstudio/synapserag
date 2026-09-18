# Arhitektonski Plan: SynapseRAG (Next-Gen Embedded RAG Engine)

## Vizija i Cilj Projekta
Razvoj profesionalnog, radikalno inovativnog RAG (Retrieval-Augmented Generation) sistema koji **nije** klasičan Python monolit (poput standardnih LangChain/LlamaIndex implementacija), već modularan, ugradiv (embeddable) engine visokih performansi dizajniran da se poveže i radi **unutar** bilo koje eksterne aplikacije, servisa, agenta ili alata.

---

## 1. Komparativna Analiza: Mainstream vs. Retki Inovativni Alati (SOTA 2025/2026)

| Dimenzija | Mainstream RAG (LangChain, LlamaIndex, Chroma) | Retki Inovativni Sistemi (HippoRAG, MemoRAG, LightRAG, ColBERT) | **SynapseRAG (Naša Nova Arhitektura)** |
| :--- | :--- | :--- | :--- |
| **Pristup Chunkingu** | Statički (fiksni prozori, npr. 512 tokena + overlap). Gubi globalni kontekst. | Late Chunking (Jina) / Contextual Chunks (Anthropic). | **Hierarchical Contextual Late-Chunking**: Globalno enkodiranje sa dinamičkim semantičkim granicama i roditeljsko-potomačkim metapodacima. |
| **Struktura Indeksa** | Samo izolovani vektori (Dense Vector Index). | HippoRAG (Personalized PageRank preko KG), LightRAG (Dual-level graf). | **Tri-Brain Index**: Fuzija Late-Interaction tenzora (ColBERT/PLAID), Neuro-asocijativnog grafa znanja (PPR) i Sparse leksičkog indeksa (BM25/SPLADE). |
| **Multi-Hop Rasuđivanje** | Loše / Skupo (zahteva višestruke spore LLM agente i skupe promptinge). | HippoRAG koristi algoritme grafa (PPR) za asocijaciju u 1 koraku. | **Neuro-Associative Spreading Activation**: Sub-sekundni grafički skokovi kroz entitete kombinovani sa vektorskim prenosom težina bez LLM latencije. |
| **Nejasni / Globalni Upiti** | Potpuni kolaps (vektori ne prepoznaju šta korisnik traži ako nema ključnih reči). | MemoRAG (System 1 generiše tragove/clues iz globalne memorije). | **Dual-System Clue Generator + Speculative Query Expander**: Brzi interni model stvara hipotetičke odgovore i navigacione signale pre same pretrage. |
| **Ugradivost (Integracija)** | Teška: masivne zavisnosti, spori serveri, monopol Python okruženja. | Uglavnom akademski skriptovi i standalone biblioteke. | **Universal Polyglot & Embeddable Core**: <br>1. In-process biblioteka (Python, Rust, TS)<br>2. Native MCP Server (Model Context Protocol za Cursor, Claude, IDE alate)<br>3. Zero-overhead IPC / gRPC sidecar. |
| **Validacija i Halucinacije** | Nepostojeća ili naivni naknadni LLM prompt. | Self-RAG / CRAG (refleksivno filtriranje). | **Deterministic Citation Circuit-Breaker**: Mapiranje na nivou tokena i karaktera sa automatskim fallback mehanizmom. |

---

## 2. Arhitektura Sistema: SynapseRAG Core

```
                           +-------------------------------------------------------------+
                           |                    HOST APLIKACIJA                          |
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
|     + Memo-Clue Predictor: Generisanje latentnih tragova za apstraktne upite                      |
|     + Multi-hop Associative Walker: Pronalaženje skrivenih veza kroz graf bez skupih LLM poziva   |
|     + Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranker                                       |
|                                                                                                   |
|  4. VERIFICATION & CITATION CIRCUIT BREAKER                                                       |
|     + Grounding & Hallucination Scorer: Provera pokrivenosti tvrdnji činjenicama                   |
|     + Exact Span Citation Mapper: Mapiranje tačnog ofseta (file:line:char) u originalu           |
|     + Adaptive Fallback Controller: Ako indeks nema odgovor -> signalizira fallback              |
|                                                                                                   |
|  5. REAL-TIME HYDRATION & CHANGE DATA CAPTURE (CDC)                                               |
|     + Incremental Graph & Vector updates bez reindeksiranja cele baze (LightRAG princip)         |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Naša Ključna Unapređenja (Šta dodajemo iznad pronađenih istraživanja)

1. **Late-Chunking + Graph Merging (Novelty 1)**:
   - Tradicionalni GraphRAG i HippoRAG odvajaju tekst na naivne parčiće pre grafičke ekstrakcije.
   - Mi povezujemo *Late Chunking* (očuvanje globalne pažnje celog dokumenta) sa *čvorovima grafa*, dajući svakom čvoru i relaciji globalni kontekstualni vektor.
2. **Personalized PageRank sa Vektorskim Usmeravanjem (Vector-Biased PPR) (Novelty 2)**:
   - Umesto običnog PPR-a iz HippoRAG-a koji koristi samo leksičke veze entiteta, naš PPR koristi kosinusnu sličnost za težine prelaza na grafu (Transition Probability Matrix je modulisan semantičkim vektorima).
3. **Embeddable Micro-Kernel sa MCP Standardom (Novelty 3)**:
   - Rešavamo najveći problem današnjih naprednih alata: nisu upotrebljivi u stvarnom softveru jer zahtevaju 10 različitih servisa (Neo4j, Milvus, Redis, Python API server).
   - SynapseRAG je dizajniran kao **jedinstvena samoodrživa biblioteka / lokalni sidecar binary** (sa ugrađenim ultra-brzim vektorskim i graf skladištem na disku poput DuckDB/SQLite/LanceDB formata) sa **izvornim MCP interfejsom** koji se povezuje u 1 liniji koda.

---

## 4. Faze Razvoja i Struktura Projekta

- **Faza 1: Skladište i Struktura Podataka (`core/storage`)**
  - Ugrađeno, samostalno hibridno skladište bez eksternih baza (Embedded Vector + Graph + Inverted index).
- **Faza 2: Ingestion & Late-Chunking Engine (`core/ingest`)**
  - Chunking uz zadržavanje globalnog konteksta i ekstrakcija entiteta/relacija.
- **Faza 3: Multi-Hop Neuro-Associative Retriever (`core/retrieval`)**
  - Vektorski vođen Personalized PageRank (PPR), ColBERT Late-Interaction MaxSim i RRF fuzija.
- **Faza 4: Query Intelligence & Clue Generator (`core/query`)**
  - Dual-system mehanizam generisanja hipoteza za nejasne i visokonivojske upite.
- **Faza 5: Ugradivi Konektori & Protokoil (`connectors/`)**
  - **Native MCP Server**: Omogućava trenutan priključak u Cursor, Claude, Windsurf, agente.
  - **In-Process Python SDK**: Za direktan import (`import synapserag`).
  - **Local Sidecar / IPC Server**: Za bilo koji drugi programski jezik (C++, Go, Node, Rust, C#).
- **Faza 6: Verifikacija i Determinističko Citiranje (`core/verify`)**
  - Hallucination detector i verifikacija činjenica sa egzaktnim fajl/red/karakter ofsetom.
