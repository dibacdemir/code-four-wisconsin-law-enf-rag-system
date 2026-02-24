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

# Common law enforcement abbreviations → expanded terms
_ABBREVIATIONS = {
    "owi": "operating while intoxicated",
    "dui": "driving under the influence",
    "dwi": "driving while intoxicated",
    "bac": "blood alcohol concentration",
    "pac": "prohibited alcohol concentration",
    "mvr": "motor vehicle record",
    "mva": "motor vehicle accident",
    "leo": "law enforcement officer",
    "tro": "temporary restraining order",
    "dv": "domestic violence",
    "cdl": "commercial driver license",
    "pts": "points",
    "fta": "failure to appear",
    "rso": "registered sex offender",
    "terry stop": "investigative stop reasonable suspicion",
    "miranda": "miranda rights warnings custodial interrogation",
    "4th amendment": "fourth amendment unreasonable search seizure",
}

# Common legal term misspellings → corrected spellings
_CORRECTIONS = {
    "suspecion": "suspicion",
    "suspicion": "suspicion",
    "probale": "probable",
    "probible": "probable",
    "consitutional": "constitutional",
    "constiutional": "constitutional",
    "constituional": "constitutional",
    "restaining": "restraining",
    "arest": "arrest",
    "arested": "arrested",
    "arrestted": "arrested",
    "mirnada": "miranda",
    "mianda": "miranda",
    "mirada": "miranda",
    "vehicel": "vehicle",
    "vehical": "vehicle",
    "trafic": "traffic",
    "traffick": "traffic",
    "reckeless": "reckless",
    "reckless": "reckless",
    "intoxicaed": "intoxicated",
    "intoxicatd": "intoxicated",
    "harasment": "harassment",
    "harrasment": "harassment",
    "assalt": "assault",
    "baterry": "battery",
    "batery": "battery",
    "burglery": "burglary",
    "robery": "robbery",
    "homocide": "homicide",
    "homocide": "homicide",
    "larcany": "larceny",
    "recless": "reckless",
    "neglagence": "negligence",
    "neglegence": "negligence",
    "warrent": "warrant",
    "warant": "warrant",
    "supena": "subpoena",
    "subpena": "subpoena",
    "witnes": "witness",
    "evidance": "evidence",
    "evidnce": "evidence",
}


def expand_query(query_text):
    """
    Apply spell correction for legal terms, then expand law enforcement
    abbreviations in the query to improve retrieval.
    E.g. 'OWI 3rd' -> 'OWI operating while intoxicated 3rd'
    E.g. 'vehicel search' -> 'vehicle search'
    """
    # Step 1: spell-correct word by word
    words = query_text.split()
    corrected_words = [_CORRECTIONS.get(w.lower(), w) for w in words]
    expanded = " ".join(corrected_words)

    # Step 2: abbreviation expansion (append expansions so original terms are kept)
    query_lower = expanded.lower()
    for abbrev, full in _ABBREVIATIONS.items():
        if abbrev in query_lower:
            if full not in query_lower:
                expanded = expanded + " " + full
    return expanded


def _follow_cross_references(collection, docs, existing_ids):
    """
    Scan retrieved doc texts for Wisconsin statute cross-references
    (patterns like '§ 940.01', 's. 346.63', 'section 940.01') and fetch
    those sections from the collection.

    Returns a list of (score, doc_text, metadata) tuples for newly found
    cross-referenced sections, each with is_cross_ref=True in metadata.
    """
    ref_patterns = [
        r"(?:§|s\.)\s*(\d{3}\.\d{2,3})",
        r"\bsec(?:tion)?\.?\s+(\d{3}\.\d{2,3})",
    ]

    found_refs = set()
    for doc in docs:
        for pattern in ref_patterns:
            for m in re.finditer(pattern, doc, re.IGNORECASE):
                found_refs.add(m.group(1))

    if not found_refs:
        return []

    cross_refs = []
    for ref in found_refs:
        try:
            cr = collection.get(
                where={"section_number": ref},
                include=["documents", "metadatas"],
            )
            for cr_doc, cr_meta in zip(
                cr.get("documents", []), cr.get("metadatas", [])
            ):
                doc_id = (
                    cr_meta.get("source_file", "")
                    + "|"
                    + cr_meta.get("section_number", "")
                )
                if doc_id not in existing_ids:
                    enriched = dict(cr_meta)
                    enriched["is_cross_ref"] = True
                    cross_refs.append((0.5, cr_doc, enriched))
                    existing_ids.add(doc_id)
        except Exception:
            continue

    return cross_refs

def hybrid_search(query_text, n_results=5, where_filter=None):
    """
    Hybrid search: combines semantic vector search with keyword matching.
    Statute numbers (e.g. '940.01', '§ 346.63') and exact terms get a boost
    when they appear literally in the retrieved chunks.
    Returns merged, re-ranked results with a confidence score (0.0 - 1.0).
    """
    query_text = expand_query(query_text)
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

    # Citation chain following: scan top docs for cross-referenced statute
    # sections and append them (up to 2 extras, deduplicated).
    existing_ids = {
        meta.get("source_file", "") + "|" + meta.get("section_number", "")
        for _, _, meta in top
    }
    cross_refs = _follow_cross_references(
        collection, [doc for _, doc, _ in top], existing_ids
    )
    top.extend(cross_refs[:2])

    # Normalize confidence to 0.0–1.0 range
    max_score = top[0][0] if top else 1.0
    confidence = round(min(top[0][0] / max(max_score, 1.0), 1.0), 3) if top else 0.0

    # Embed per-source similarity score into each metadata dict so it flows to the frontend
    enriched_metas = []
    for final_score, doc, meta in top:
        enriched = dict(meta)
        enriched["similarity_score"] = round(min(final_score, 1.0), 3)
        enriched_metas.append(enriched)

    return {
        "documents": [[item[1] for item in top]],
        "metadatas": [enriched_metas],
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