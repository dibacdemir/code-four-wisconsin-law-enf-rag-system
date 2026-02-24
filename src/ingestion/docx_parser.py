from docx import Document
import os


def parse_docx(file_path):
    filename = os.path.basename(file_path)
    doc = Document(file_path)

    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    return [{
        "text": "\n".join(full_text),
        "page_number": 1,
        "source_file": filename,
        "doc_type": "department_policy",
    }]