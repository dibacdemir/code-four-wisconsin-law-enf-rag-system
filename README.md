# Wisconsin Law Enforcement Legal Chat — RAG System

A proof-of-concept Retrieval-Augmented Generation (RAG) system enabling Wisconsin law enforcement officers to query state statutes, case law, and department policies through a conversational interface.

---

## Architecture

```
PDFs (statutes, case law)
        │
        ▼
  [PDF Parser]           → extracts text + classifies doc type
        │
        ▼
  [Chunker]              → statute-aware sectioning or paragraph splitting
        │
        ▼
  [ChromaDB]             → OpenAI embeddings (text-embedding-3-small), cosine similarity
        │
  [Hybrid Search]        → semantic score + keyword boost for statute refs
        │
        ▼
  [GPT-4o-mini]          → grounded response with source citations
        │
        ▼
  [FastAPI + Web UI]     → chat interface with confidence scores
```

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Vector DB | ChromaDB | Local persistence, no infra required, easy metadata filtering |
| Embeddings | OpenAI `text-embedding-3-small` | Strong legal text performance, cost-efficient |
| LLM | OpenAI GPT-4o-mini | Low latency, sufficient reasoning for statute Q&A |
| Framework | FastAPI | Async-ready, auto-generated docs at `/docs` |
| Frontend | Vanilla HTML/JS | Zero build step, easy to demo and modify |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd code-four-wisconsin-law-enf-rag-system
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here   # optional
PINECONE_API_KEY=your_pinecone_key_here     # optional
```

### 3. Add legal documents

Place PDF files in `data/raw/`. Naming conventions:
- Wisconsin statutes: `<chapter>.pdf` (e.g. `940.pdf`, `346.pdf`)
- Case law: any filename containing `case` or `opinion`
- Department policies: any filename containing `policy`

### 4. Run ingestion

```bash
python ingest.py --reset
```

This parses all PDFs, chunks them, generates embeddings, and stores them in ChromaDB. Re-run with `--reset` whenever you add new documents.

### 5. Start the API

```bash
python src/api/main.py
```

Server runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### 6. Open the frontend

Open `frontend/index.html` directly in your browser. No build step required.

---

## Running Tests

```bash
venv/bin/python -m pytest tests/ -v
```

17 tests covering: document classification, chunking, hybrid search, confidence scoring, and all API endpoints.

---

## Project Structure

```
├── data/raw/               # Source PDFs (statutes, case law)
├── src/
│   ├── api/main.py         # FastAPI server
│   ├── ingestion/
│   │   ├── pdf_parser.py   # PDF text extraction and classification
│   │   └── chunker.py      # Statute-aware and case law chunking
│   ├── retrieval/
│   │   └── vector_store.py # ChromaDB + hybrid search
│   └── generation/
│       ├── llm_client.py   # OpenAI GPT-4o-mini response generation
│       └── prompts.py      # System prompt and context builder
├── frontend/index.html     # Web chat interface
├── tests/test_core.py      # Unit tests
├── ingest.py               # Ingestion pipeline entry point
└── chroma_db/              # Persistent vector store (auto-created)
```

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /query`
```json
{
  "question": "What are the elements of OWI 3rd offense in Wisconsin?",
  "doc_type_filter": "statute"   // optional: "statute" | "case_law"
}
```
Returns:
```json
{
  "answer": "...",
  "sources": [{"source_file": "346.pdf", "section_number": "346.63", ...}],
  "confidence": 0.82
}
```

---

## Example Queries

- *"What are the elements required for OWI 3rd offense in Wisconsin?"*
- *"Can I search a vehicle during a traffic stop without consent?"*
- *"What's the statute of limitations for misdemeanor theft?"*
- *"Show me recent cases about Terry stops in Wisconsin"*
- *"What Miranda warnings are required for juveniles?"*

---

## Design Decisions

**Why ChromaDB over Pinecone?** Local persistence means zero network latency and no API costs during development. For production scale, swap `get_vector_store()` in `vector_store.py` to point at Pinecone — the interface is identical.

**Why hybrid search?** Statute references like `§ 940.01` are exact strings, not semantic concepts. Pure embedding search misses exact citations. The keyword boost layer adds `+0.15` per matched statute number and `+0.05` per significant keyword, capped at `+0.4` to prevent keyword results from drowning out semantically relevant chunks.

**Why GPT-4o-mini over GPT-4o?** Response latency matters in field use. GPT-4o-mini handles statute Q&A well at ~3x lower cost and faster response times.

**Chunking strategy:** Statutes are split on section boundaries (e.g. `940.01`, `346.63`) rather than fixed character counts, preserving legal completeness. Case law uses overlapping paragraph splits (1000 chars, 200 overlap) to avoid cutting across reasoning chains.
