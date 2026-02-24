"""
Unit tests for critical RAG pipeline components.
Run with: pytest tests/ -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()


# ─── PDF Parser Tests ─────────────────────────────────────────────────────────

from ingestion.pdf_parser import classify_document

def test_classify_statute_by_number():
    assert classify_document("940.pdf") == "statute"
    assert classify_document("346.pdf") == "statute"

def test_classify_case_law():
    assert classify_document("case_opinion_1.pdf") == "case_law"
    assert classify_document("opinion_smith_v_doe.pdf") == "case_law"

def test_classify_policy():
    assert classify_document("policy_pursuit_driving.pdf") == "department_policy"

def test_classify_unknown():
    assert classify_document("random_document.pdf") == "other"


# ─── Chunker Tests ────────────────────────────────────────────────────────────

from ingestion.chunker import chunk_statute, chunk_case_law, chunk_documents

SAMPLE_STATUTE_PAGES = [
    {
        "text": "346.63 Operating while intoxicated.\n(1) No person may drive a vehicle...\n346.64 Reckless driving.\n(1) No person may drive a vehicle...",
        "page_number": 1,
        "source_file": "346.pdf",
        "doc_type": "statute",
    }
]

SAMPLE_CASE_PAGES = [
    {
        "text": "The court held that the officer had reasonable suspicion.\n\nThe defendant argued that the stop was unlawful.",
        "page_number": 1,
        "source_file": "case_opinion_1.pdf",
        "doc_type": "case_law",
    }
]

def test_chunk_statute_returns_chunks():
    chunks = chunk_statute(SAMPLE_STATUTE_PAGES)
    assert len(chunks) > 0

def test_chunk_statute_has_required_metadata():
    chunks = chunk_statute(SAMPLE_STATUTE_PAGES)
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "source_file" in chunk["metadata"]
        assert "doc_type" in chunk["metadata"]
        assert "chapter" in chunk["metadata"]

def test_chunk_case_law_returns_chunks():
    chunks = chunk_case_law(SAMPLE_CASE_PAGES)
    assert len(chunks) > 0

def test_chunk_case_law_has_required_metadata():
    chunks = chunk_case_law(SAMPLE_CASE_PAGES)
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["doc_type"] == "case_law"

def test_chunk_documents_routes_correctly():
    all_pages = SAMPLE_STATUTE_PAGES + SAMPLE_CASE_PAGES
    chunks = chunk_documents(all_pages)
    doc_types = {c["metadata"]["doc_type"] for c in chunks}
    assert "statute" in doc_types
    assert "case_law" in doc_types


# ─── Hybrid Search Tests ──────────────────────────────────────────────────────

from retrieval.vector_store import hybrid_search

def test_hybrid_search_returns_expected_keys():
    results = hybrid_search("OWI third offense Wisconsin", n_results=3)
    assert "documents" in results
    assert "metadatas" in results
    assert "confidence" in results

def test_hybrid_search_respects_n_results():
    results = hybrid_search("vehicle search traffic stop", n_results=3)
    # Citation chain following may append up to 2 cross-referenced sections
    assert len(results["documents"][0]) <= 5

def test_hybrid_search_confidence_in_range():
    results = hybrid_search("Miranda rights juvenile", n_results=3)
    assert 0.0 <= results["confidence"] <= 1.0

def test_hybrid_search_doc_type_filter():
    results = hybrid_search("criminal penalty", n_results=3, where_filter={"doc_type": "statute"})
    for meta in results["metadatas"][0]:
        assert meta["doc_type"] == "statute"


# ─── API Tests ────────────────────────────────────────────────────────────────

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_returns_answer_and_sources():
    response = client.post("/query", json={"question": "What is OWI in Wisconsin?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert len(data["answer"]) > 0

def test_query_rejects_empty_question():
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422  # Pydantic validation error

def test_query_with_doc_type_filter():
    response = client.post("/query", json={
        "question": "traffic stop rules",
        "doc_type_filter": "statute"
    })
    assert response.status_code == 200
