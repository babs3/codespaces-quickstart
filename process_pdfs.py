import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Configuration
PDF_FOLDER = "materials/pdfs"
INDEX_PATH = "vector_store/faiss_index"
DATA_PATH = "vector_store/documents.pkl"
CHUNK_SIZE = 300  # Adjust the chunk size to control granularity

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract and split text from PDF
def extract_text_chunks(pdf_path, chunk_size=CHUNK_SIZE):
    doc = fitz.open(pdf_path)
    full_text = " ".join([page.get_text("text") for page in doc])

    # Split text into chunks of CHUNK_SIZE words
    words = full_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    return chunks

# Process all PDFs and create document chunks
documents = []
metadata = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)
        text_chunks = extract_text_chunks(pdf_path)
        documents.extend(text_chunks)
        metadata.extend([{"file": file, "chunk_id": i} for i in range(len(text_chunks))])

# Generate embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Save document chunks and metadata
with open(DATA_PATH, "wb") as f:
    pickle.dump((documents, metadata), f)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

print("Knowledge base has been created with chunked documents.")
