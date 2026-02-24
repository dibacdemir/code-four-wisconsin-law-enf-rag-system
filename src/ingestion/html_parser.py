from html.parser import HTMLParser
from ingestion.pdf_parser import classify_document
import os



class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.text.append(data.strip())


def parse_html(file_path):
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    extractor = TextExtractor()
    extractor.feed(content)

    doc_type = classify_document(filename)
    # HTML files from Wisconsin Legislature are statutes unless filename says otherwise
    if doc_type == "other":
        doc_type = "statute"

    return [{
        "text": "\n".join(extractor.text),
        "page_number": 1,
        "source_file": filename,
        "doc_type": doc_type,
    }]
