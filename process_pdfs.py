import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
import os
import fitz  # PyMuPDF
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
PDF_FOLDER = "materials/GEE_pdfs"
VECTOR_DB_PATH = "vector_store"  # Path for storing the vector database

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(name="class_materials")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_by_page(pdf_path):
    """Extracts text from each page of a PDF separately."""
    doc = fitz.open(pdf_path)
    page_chunks = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if text:  # Ensure non-empty text
            page_chunks.append({"text": text, "page": page_num + 1})
    
    return page_chunks

# Process all PDFs
documents = []
metadata = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)
        page_chunks = extract_text_by_page(pdf_path)

        for chunk in page_chunks:
            documents.append(chunk["text"])
            metadata.append({"file": file, "page": chunk["page"]})

# Generate embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Store in ChromaDB
for i, doc_text in enumerate(documents):
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        metadatas=[metadata[i]],
        documents=[doc_text]
    )

print("Knowledge base has been created with page-based document storage.")
