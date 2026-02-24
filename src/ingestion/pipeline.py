import os
from ingestion.pdf_parser import parse_pdf
from ingestion.docx_parser import parse_docx
from ingestion.html_parser import parse_html
from ingestion.pdf_parser import classify_document


def parse_all_documents(directory):
    """
    Parse all supported document types in a directory.
    Routes each file to the appropriate parser.
    """
    all_pages = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.endswith(".pdf"):
            print(f"Parsing PDF: {filename}")
            pages = parse_pdf(file_path)
        elif filename.endswith(".docx"):
            print(f"Parsing DOCX: {filename}")
            pages = parse_docx(file_path)
        elif filename.endswith(".html") or filename.endswith(".htm"):
            print(f"Parsing HTML: {filename}")
            pages = parse_html(file_path)
        elif filename.endswith(".txt"):
            print(f"Parsing TXT: {filename}")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                pages = [{
                    "text": f.read(),
                    "page_number": 1,
                    "source_file": filename,
                    "doc_type": classify_document(filename),
                }]
        else:
            print(f"Skipping unsupported format: {filename}")
            continue

        print(f"  → Extracted {len(pages)} pages")
        all_pages.extend(pages)

    return all_pages