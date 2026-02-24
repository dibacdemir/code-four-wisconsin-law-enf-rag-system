import os
import re 
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


def get_vector_store():
    """
    Initialize ChromaDB with OpenAI embeddings.
    Returns the collection.
    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    client = chromadb.PersistentClient(path=os.path.join(project_root, "chroma_db"))

    collection = client.get_or_create_collection(
        name="wisconsin_legal",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def index_chunks(chunks):
    """
    Add chunks to the vector store with metadata.
    """
    collection = get_vector_store()

    # ChromaDB needs lists of ids, documents, and metadatas
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    # ChromaDB has a batch limit, so insert in batches of 100
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed batch {start // batch_size + 1} ({start}-{min(end, len(ids))})")

    print(f"\nTotal documents in collection: {collection.count()}")
    return collection


def query_vector_store(query_text, n_results=5, where_filter=None):
    """
    Search the vector store for relevant chunks.
    """
    collection = get_vector_store()

    kwargs = {
        "query_texts": [query_text],
        "n_results": n_results,
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)
    return results

def hybrid_search(query_text, n_results=5, where_filter=None):
    """
    Hybrid search: combines semantic vector search with keyword matching.
    Statute numbers (e.g. '940.01', '§ 346.63') and exact terms get a boost
    when they appear literally in the retrieved chunks.
    Returns merged, re-ranked results with a confidence score (0.0 - 1.0).
    """
    collection = get_vector_store()

    # Fetch more candidates than needed so we have room to re-rank
    fetch_n = min(n_results * 3, collection.count())

    kwargs = {
        "query_texts": [query_text],
        "n_results": fetch_n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # Extract statute-style patterns from the query (e.g. "940.01", "346.63")
    statute_refs = re.findall(r"\b\d{3}\.\d{2,3}\b", query_text)

    # Also extract significant words (4+ chars, not stopwords) for keyword boosting
    stopwords = {"what", "when", "where", "which", "that", "this", "with", "from",
                 "have", "does", "are", "the", "for", "and", "can", "during"}
    keywords = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", query_text)
                if w.lower() not in stopwords]

    scored = []
    for doc, meta, distance in zip(docs, metas, distances):
        # Semantic score: ChromaDB cosine distance → similarity (lower distance = better)
        semantic_score = 1.0 - distance

        # Keyword boost: +0.15 per statute ref found, +0.05 per keyword found
        keyword_score = 0.0
        doc_lower = doc.lower()
        for ref in statute_refs:
            if ref in doc:
                keyword_score += 0.15
        for kw in keywords:
            if kw in doc_lower:
                keyword_score += 0.05

        # Cap keyword boost at 0.4 so it can't completely override semantic score
        keyword_score = min(keyword_score, 0.4)

        final_score = semantic_score + keyword_score
        scored.append((final_score, doc, meta))

    # Sort by final score descending, take top n_results
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_results]

    # Normalize confidence to 0.0–1.0 range
    max_score = top[0][0] if top else 1.0
    confidence = round(min(top[0][0] / max(max_score, 1.0), 1.0), 3) if top else 0.0

    return {
        "documents": [[item[1] for item in top]],
        "metadatas": [[item[2] for item in top]],
        "distances": [[1.0 - item[0] for item in top]],
        "confidence": confidence,
    }


# Quick test: index all chunks and run a sample query
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.pipeline import parse_all_documents
    from ingestion.chunker import chunk_documents

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    data_dir = os.path.normpath(data_dir)

    # Parse and chunk
    pages = parse_all_documents(data_dir)
    chunks = chunk_documents(pages)

    # Index
    print("\nIndexing chunks into ChromaDB...")
    index_chunks(chunks)

    # Test query
    print("\n--- Test Query: 'vehicle search during traffic stop' ---")
    results = query_vector_store("vehicle search during traffic stop")

    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\nResult {i+1} [{meta}]:")
        print(doc[:200])