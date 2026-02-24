# System Architecture — Wisconsin Law Enforcement Legal Chat RAG

## RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                      │
│                    (run once via ingest.py)                  │
│                                                              │
│  data/raw/*.pdf                                              │
│       │                                                      │
│       ▼                                                      │
│  pdf_parser.py                                               │
│  • Extracts text page by page using pdfplumber               │
│  • Classifies doc type by filename pattern:                  │
│    - /^\d{3}\.pdf$/ → statute                               │
│    - "case|opinion"  → case_law                              │
│    - "policy"        → department_policy                     │
│       │                                                      │
│       ▼                                                      │
│  chunker.py                                                  │
│  • Statutes: split on section boundaries (e.g. 940.01)       │
│    Falls back to 1000-char chunks with 200-char overlap      │
│  • Case law: paragraph splits, 1000/200 overlap              │
│  • Each chunk carries metadata: source_file, doc_type,       │
│    chapter, section_number                                   │
│       │                                                      │
│       ▼                                                      │
│  ChromaDB (chroma_db/)                                       │
│  • Embedding model: text-embedding-3-small (OpenAI)          │
│  • Distance metric: cosine similarity                        │
│  • 1,972 chunks indexed from 7 source documents              │
└─────────────────────────────────────────────────────────────┘z

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                   (per request via API)                      │
│                                                              │
│  User Question                                               │
│       │                                                      │
│       ▼                                                      │
│  FastAPI /query endpoint                                     │
│  • Validates: non-empty, optional doc_type_filter            │
│       │                                                      │
│       ▼                                                      │
│  hybrid_search() in vector_store.py                          │
│  • Fetches 15 candidates via semantic search (ChromaDB)      │
│  • Extracts statute refs from query (e.g. "940.01")          │
│  • Keyword boost: +0.15 per statute match, +0.05 per term    │
│  • Re-ranks all 15, returns top 5 with confidence score      │
│       │                                                      │
│       ▼                                                      │
│  get_llm_response() via GPT-4o-mini                          │
│  • System prompt enforces: cite sources, no outside          │
│    knowledge, include legal disclaimer                       │
│  • Context includes chunk text + metadata for each source    │
│       │                                                      │
│       ▼                                                      │
│  Response: answer + sources + confidence (0.0–1.0)           │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Vector Database: ChromaDB (local) over Pinecone (cloud)
ChromaDB persists to SQLite on disk with zero infrastructure. For this proof-of-concept, this eliminates network latency and API costs during development. Migrating to Pinecone for production requires only changing `get_vector_store()` — the query interface is identical.

### Embedding Model: text-embedding-3-small
Benchmarked well on legal domain text retrieval tasks. At 1536 dimensions it's significantly cheaper than `text-embedding-3-large` with minimal retrieval quality loss for statute-level queries.

### LLM: GPT-4o-mini over GPT-4o
Field use demands low latency. GPT-4o-mini handles statute Q&A correctly at ~3x lower cost and ~2x faster response times. The strict system prompt (cite sources, no outside knowledge) compensates for reduced reasoning capacity.

### Hybrid Search over Pure Semantic Search
Statute references like `§ 940.01` are exact strings. Embedding-based search treats them as semantic concepts, which can miss exact citation lookups. The keyword boost layer rescores candidates after retrieval without a second embedding call, keeping latency low.

### Chunking by Statute Section
Fixed-size character chunking would split section `940.01(1)(a)` from `940.01(1)(b)`, producing incomplete legal context. Splitting on section boundaries preserves subsection groupings, which matters when officers ask about specific elements of an offense.

## Scalability Considerations

| Concern | Current Approach | Production Path |
|---|---|---|
| Vector store | ChromaDB SQLite | Pinecone or Weaviate with namespaces per jurisdiction |
| Document volume | 7 PDFs, 1,972 chunks | Add `--data-dir` path to ingest.py, re-run |
| Concurrency | Single FastAPI worker | `uvicorn --workers 4` or deploy behind gunicorn |
| Embedding cost | Per-query via OpenAI | Cache embeddings for repeated queries |
| LLM cost | GPT-4o-mini | Fine-tune on WI legal corpus, self-host Mistral/Llama |

## Security & Privacy

- **API keys** stored in `.env`, excluded from version control via `.gitignore`
- **No PII stored** — ChromaDB contains only statute text and case law, no officer data
- **CORS** currently open (`*`) — in production, restrict to known frontend origins
- **No authentication** on API — production deployment should add JWT or API key middleware
- **Legal disclaimer** surfaced in every response and in the UI banner
- **Use-of-force queries** handled by system prompt: requires citation, flags when policy-specific

## Document Corpus

| File | Type | Coverage |
|---|---|---|
| 346.pdf | Statute | Ch. 346 — Vehicle & Traffic Law |
| 940.pdf | Statute | Ch. 940 — Crimes Against Life & Body |
| 968.pdf | Statute | Ch. 968 — Criminal Procedure |
| 971.pdf | Statute | Ch. 971 — Criminal Court Procedure |
| case_opinion_1.pdf | Case Law | Wisconsin appellate opinion |
| case_opinion_2.pdf | Case Law | Wisconsin appellate opinion |
| case_opinion_3.pdf | Case Law | Wisconsin appellate opinion |
