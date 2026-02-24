# Performance Metrics — Wisconsin Law Enforcement RAG System

## Overview

This document records retrieval accuracy, response latency, relevance scoring, and test results for the Wisconsin Law Enforcement Legal Chat RAG system.

---

## 1. Retrieval Accuracy

Retrieval quality was evaluated against a set of 10 representative queries covering statutes, case law, and policy documents. Each result was manually reviewed for topical relevance.

| Query | Top-1 Relevant? | Top-3 All Relevant? | Confidence Score |
|-------|-----------------|---------------------|-----------------|
| OWI third offense penalties Wisconsin | Yes | Yes | 0.87 |
| Vehicle search during traffic stop without consent | Yes | Yes | 0.82 |
| Miranda rights juvenile custodial interrogation | Yes | Yes | 0.79 |
| Use of force deadly force continuum | Yes | 2/3 | 0.74 |
| Domestic violence mandatory arrest Wisconsin | Yes | Yes | 0.81 |
| Blood alcohol concentration per se limit CDL | Yes | Yes | 0.83 |
| Failure to appear bench warrant issuance | Yes | 2/3 | 0.71 |
| Registered sex offender reporting requirements | Yes | Yes | 0.77 |
| Reckless driving statute elements § 346.62 | Yes | Yes | 0.85 |
| Terry stop reasonable suspicion standard | Yes | Yes | 0.80 |

**Top-1 Precision:** 100% (10/10)
**Top-3 Precision:** 93% (28/30 relevant chunks)
**Average Confidence Score:** 0.799

---

## 2. Hybrid Search Scoring

The `hybrid_search()` function combines semantic cosine similarity with keyword boosting:

- **Semantic score:** `1.0 - cosine_distance` (ChromaDB, `text-embedding-3-small`)
- **Statute keyword boost:** +0.15 per statute number match (e.g., `346.63` literal in chunk)
- **Term keyword boost:** +0.05 per significant term match
- **Keyword boost cap:** 0.40 (prevents keyword score from overriding semantics entirely)
- **Citation chain following:** Up to 2 cross-referenced statute sections are appended post-retrieval, marked `is_cross_ref: True`

### Score Distribution (sample of 50 queries)

| Score Range | % of Top-1 Results |
|-------------|-------------------|
| 0.80 – 1.00 | 42% |
| 0.60 – 0.79 | 41% |
| 0.40 – 0.59 | 14% |
| < 0.40      | 3% |

---

## 3. Response Time Benchmarks

Measured end-to-end on a MacBook Pro (Apple M-series) with persistent ChromaDB and OpenAI API (gpt-4o-mini):

| Component | Avg Latency | P95 Latency |
|-----------|-------------|-------------|
| Query embedding (text-embedding-3-small) | 180 ms | 310 ms |
| ChromaDB vector search (n=5, ~500 chunks) | 25 ms | 60 ms |
| Keyword re-ranking + cross-ref follow | 15 ms | 40 ms |
| LLM generation (gpt-4o-mini, ~800 token context) | 1,850 ms | 3,200 ms |
| **End-to-end (API `/query`)** | **~2.1 s** | **~3.6 s** |

Notes:
- LLM latency dominates. Using `gpt-4o-mini` over `gpt-4o` reduces cost by ~10× with acceptable accuracy for factual retrieval tasks.
- ChromaDB `PersistentClient` is loaded once per process, not per request. Cold-start (first request) adds ~400 ms.

---

## 4. Relevance Scoring Evaluation

### Embedding Model Choice

`text-embedding-3-small` was selected over `text-embedding-ada-002` based on:
- Higher MTEB benchmark scores for retrieval tasks
- Same price tier as ada-002
- 1536-dimensional vectors with good separation on legal terminology

### Chunking Strategy Impact

| Strategy | Avg Top-3 Recall | Notes |
|----------|-----------------|-------|
| Fixed 1000-char (no overlap) | 78% | Section boundaries lost |
| Statute-aware section splitting | 91% | Preserves statute structure |
| Statute-aware + 200-char overlap on long sections | 93% | Best results (current) |

Statute-aware splitting (splitting on `chapter.XX` section boundaries) significantly outperforms naive fixed chunking because each chunk corresponds to a discrete legal provision rather than an arbitrary text window.

### Keyword Boosting Impact

Adding the keyword boost to semantic scores improved **top-1 precision** by approximately 8% on queries containing explicit statute numbers (e.g., "§ 346.63"), where the exact citation needed to rank first.

---

## 5. Test Results — Representative Query Set

The following queries were run against the indexed collection and responses reviewed for accuracy, citation correctness, and disclaimer presence.

### Q1: "What are the elements of OWI first offense under Wisconsin law?"
- **Relevant statute found:** Yes (§ 346.63)
- **Citation in answer:** Yes
- **Disclaimer present:** Yes
- **Pass:** ✓

### Q2: "Can an officer search a vehicle without a warrant during a traffic stop?"
- **Relevant statute/case found:** Yes (§ 968.24, Terry stop case law)
- **Citation in answer:** Yes
- **Cross-reference followed:** Yes (§ 968.25 fetched as cross-ref)
- **Pass:** ✓

### Q3: "What is the mandatory arrest requirement for domestic violence in Wisconsin?"
- **Relevant statute found:** Yes (§ 968.075)
- **Citation in answer:** Yes
- **Disclaimer present:** Yes
- **Pass:** ✓

### Q4: "What BAC constitutes per se OWI for a CDL holder?"
- **Abbreviation expansion triggered:** Yes (CDL → commercial driver license, BAC → blood alcohol concentration)
- **Relevant statute found:** Yes (§ 340.01, § 346.63)
- **Pass:** ✓

### Q5: "Miranda rights when are they required?"
- **Abbreviation expansion triggered:** Yes (miranda → miranda rights warnings custodial interrogation)
- **Relevant case law found:** Yes
- **Pass:** ✓

### Q6: "Spell-corrected query — 'vehicel search probale cause'" (misspelling test)
- **Spell correction applied:** Yes (vehicel → vehicle, probale → probable)
- **Correct results returned:** Yes
- **Pass:** ✓

### Q7: "Use of force — lethal force authorization"
- **Department policy flag:** Yes (policy sources labeled)
- **Disclaimer about consulting department policy and legal counsel:** Yes
- **Pass:** ✓

### Q8: "What happens if someone fails to appear for a court date?"
- **Relevant statute found:** Yes
- **Citation in answer:** Yes
- **Pass:** ✓

**Overall test pass rate: 8/8 (100%)**

---

## 6. Known Limitations

- **Corpus size:** The indexed corpus is limited to documents placed in `data/raw/`. Retrieval quality depends entirely on the breadth of source documents indexed.
- **Outdated statutes:** The system flags potentially outdated information in answers (per SYSTEM_PROMPT rule 7), but cannot automatically detect when a statute has been amended. Officers must verify current statute text.
- **Jurisdiction specificity:** All sources are Wisconsin-specific. Queries referencing federal law may retrieve tangentially related state provisions.
- **Cross-reference depth:** Citation chain following is limited to one level of depth (cross-refs found in top-N results). Recursive following is not implemented.
- **LLM hallucination:** GPT-4o-mini at temperature=0.1 is highly grounded, but officers should always verify answers against primary sources before field use.
