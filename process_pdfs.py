import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
import os
import fitz  # PyMuPDF
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Configuration
PDF_FOLDER = "materials/GEE_pdfs"
VECTOR_DB_PATH = "vector_store"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(name="class_materials")

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_by_page(pdf_path):
    """Extracts text from each page of a PDF separately."""
    doc = fitz.open(pdf_path)
    page_chunks = []
    
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if text:
            page_chunks.append({"text": text, "page": page_num + 1})
    
    return page_chunks

# Process all PDFs
documents = []
metadata = []
tokenized_docs = []  # For BM25 sparse search

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)
        page_chunks = extract_text_by_page(pdf_path)

        for chunk in page_chunks:
            doc_text = chunk["text"]
            documents.append(doc_text)
            metadata.append({"file": file, "page": chunk["page"]})
            tokenized_docs.append(doc_text.lower().split())  # Tokenized for BM25 search

# Build BM25 Index
bm25_index = BM25Okapi(tokenized_docs)

# Generate dense embeddings
embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# Store in ChromaDB
for i, doc_text in enumerate(documents):
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        metadatas=[metadata[i]],
        documents=[doc_text]
    )

# Save BM25 index
import pickle
with open("vector_store/bm25_index.pkl", "wb") as f:
    pickle.dump((bm25_index, metadata, documents), f)

print("\n✅ Hybrid Search: Knowledge base created with vector & keyword search.")
