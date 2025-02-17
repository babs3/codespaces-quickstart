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
CHUNK_SIZE = 200  # Number of words per chunk
OVERLAP = 50  # Number of words to overlap between chunks

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(name="class_materials")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_with_sliding_window(pdf_path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Extracts text from a PDF using a sliding window approach."""
    doc = fitz.open(pdf_path)
    text_chunks = []
    metadata = []

    for page_num in range(len(doc)):
        full_text = doc[page_num].get_text("text").strip()
        words = full_text.split()

        # Create sliding window chunks
        for i in range(0, len(words), chunk_size - overlap):  # Overlapping chunks
            chunk = " ".join(words[i:i + chunk_size])
            text_chunks.append(chunk)
            metadata.append({"file": os.path.basename(pdf_path), "page": page_num + 1})

    return text_chunks, metadata

# Process all PDFs
documents = []
metadata = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)
        page_chunks, page_metadata = extract_text_with_sliding_window(pdf_path)

        documents.extend(page_chunks)
        metadata.extend(page_metadata)

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

print("✅ Knowledge base has been created using sliding window chunking.")
