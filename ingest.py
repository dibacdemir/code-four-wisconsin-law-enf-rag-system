#!/usr/bin/env python3
"""
Ingestion pipeline: parse PDFs → chunk → index into ChromaDB.

Usage:
    python ingest.py               # index all PDFs in data/raw/
    python ingest.py --reset       # wipe ChromaDB first, then re-index
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from ingestion.pipeline import parse_all_documents
from ingestion.chunker import chunk_documents
from retrieval.vector_store import get_vector_store, index_chunks


def reset_collection():
    """Delete and recreate the ChromaDB collection (clean slate)."""
    import chromadb
    project_root = os.path.dirname(__file__)
    client = chromadb.PersistentClient(path=os.path.join(project_root, "chroma_db"))
    try:
        client.delete_collection("wisconsin_legal")
        logger.info("Deleted existing 'wisconsin_legal' collection.")
    except Exception:
        logger.info("No existing collection to delete.")


def main():
    parser = argparse.ArgumentParser(description="Ingest Wisconsin legal PDFs into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Wipe ChromaDB before indexing.")
    parser.add_argument("--data-dir", default="data/raw", help="Path to directory of PDFs.")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)

    if not os.path.isdir(data_dir):
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    if args.reset:
        reset_collection()

    # Step 1: Parse
    logger.info("=== Step 1: Parsing documents from %s ===", data_dir)
    pages = parse_all_documents(data_dir)
    logger.info("Total pages extracted: %d", len(pages))

    if not pages:
        logger.error("No pages extracted. Check that PDFs exist in %s", data_dir)
        sys.exit(1)

    # Step 2: Chunk
    logger.info("=== Step 2: Chunking documents ===")
    chunks = chunk_documents(pages)
    logger.info("Total chunks created: %d", len(chunks))

    # Step 3: Index
    logger.info("=== Step 3: Indexing into ChromaDB ===")
    index_chunks(chunks)
    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    main()
