import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_statute(pages):
    """
    Chunk statute documents by section number (e.g., 940.01, 346.63).
    Falls back to character splitting for long sections.
    """
    # Combine all pages into one text block per document
    full_text = "\n".join([p["text"] for p in pages])
    source_file = pages[0]["source_file"]
    doc_type = pages[0]["doc_type"]

    # Extract chapter number from filename (e.g., "940" from "940.pdf")
    chapter = source_file.replace(".pdf", "")

    # Split on statute section boundaries
    # Pattern matches things like "940.01" or "346.63" at the start of a line or after whitespace
    section_pattern = rf"(?=(?:^|\n)\s*{chapter}\.\d{{2,3}})"
    sections = re.split(section_pattern, full_text)

    chunks = []
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Try to extract the section number
        section_match = re.match(rf"({chapter}\.\d{{2,3}})", section)
        section_number = section_match.group(1) if section_match else "unknown"

        metadata = {
            "source_file": source_file,
            "doc_type": doc_type,
            "chapter": chapter,
            "section_number": section_number,
        }

        # If section is short enough, keep it as one chunk
        if len(section) <= 1200:
            chunks.append({"text": section, "metadata": metadata})
        else:
            # Split long sections but keep metadata
            sub_chunks = fallback_splitter.split_text(section)
            for i, sub in enumerate(sub_chunks):
                chunk_metadata = {**metadata, "sub_chunk": i + 1}
                chunks.append({"text": sub, "metadata": chunk_metadata})

    return chunks


def chunk_case_law(pages):
    """
    Chunk court opinions by paragraphs with overlap.
    """
    full_text = "\n".join([p["text"] for p in pages])
    source_file = pages[0]["source_file"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    text_chunks = splitter.split_text(full_text)

    chunks = []
    for i, text in enumerate(text_chunks):
        chunks.append({
            "text": text,
            "metadata": {
                "source_file": source_file,
                "doc_type": "case_law",
                "chunk_index": i + 1,
            },
        })

    return chunks


def chunk_documents(all_pages):
    """
    Route pages to the appropriate chunking strategy based on doc_type.
    """
    # Group pages by source file
    docs = {}
    for page in all_pages:
        key = page["source_file"]
        if key not in docs:
            docs[key] = []
        docs[key].append(page)

    all_chunks = []
    for source_file, pages in docs.items():
        doc_type = pages[0]["doc_type"]
        print(f"Chunking: {source_file} ({doc_type})")

        if doc_type == "statute":
            chunks = chunk_statute(pages)
        else:
            chunks = chunk_case_law(pages)

        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


# Quick test
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.pdf_parser import parse_all_pdfs

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    data_dir = os.path.normpath(data_dir)

    pages = parse_all_pdfs(data_dir)
    chunks = chunk_documents(pages)

    print(f"\nTotal chunks: {len(chunks)}")

    # Show a few sample chunks
    for chunk in chunks[:3]:
        print(f"\n--- {chunk['metadata']} ---")
        print(chunk["text"][:300])