import os
from dotenv import load_dotenv

load_dotenv()

print("OpenAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Anthropic key loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))

import chromadb
client = chromadb.Client()
print("ChromaDB working:", bool(client))

import pdfplumber
print("pdfplumber working: True")

print("\nAll good! Ready for Phase 1.")
