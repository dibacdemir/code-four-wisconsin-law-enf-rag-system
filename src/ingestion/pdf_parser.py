import pdfplumber
import os
import re


def parse_pdf(file_path):
    """
    Extract text and metadata from a PDF file.
    Returns a list of dicts, one per page.
    """
    filename = os.path.basename(file_path)
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": i + 1,
                    "source_file": filename,
                    "doc_type": classify_document(filename),
                })

    return pages


def classify_document(filename):
    """
    Classify document type based on filename.
    """
    # Statute chapters are named like 346.pdf, 940.pdf, etc.
    if re.match(r"^\d{3}\.pdf$", filename):
        return "statute"
    elif "case" in filename.lower() or "opinion" in filename.lower():
        return "case_law"
    elif "policy" in filename.lower():
        return "department_policy"
    else:
        return "other"


def parse_all_pdfs(directory):
    """
    Parse all PDFs in a directory.
    Returns a flat list of all pages from all documents.
    """
    all_pages = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            print(f"Parsing: {filename}")
            pages = parse_pdf(file_path)
            print(f"  → Extracted {len(pages)} pages")
            all_pages.extend(pages)
    return all_pages


# Quick test to see output
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    data_dir = os.path.normpath(data_dir)

    pages = parse_all_pdfs(data_dir)
    print(f"\nTotal pages extracted: {len(pages)}")

    # Print first page of first document to inspect the text
    if pages:
        print(f"\n--- Sample from {pages[0]['source_file']} (page {pages[0]['page_number']}) ---")
        print(f"Doc type: {pages[0]['doc_type']}")
        print(pages[0]["text"][:500])