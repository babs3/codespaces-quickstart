import os
import fitz  # PyMuPDF
import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
PDF_FOLDER = "materials/pdfs"
CHUNK_SIZE = 80  # Adjust the chunk size to control granularity

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = chroma_client.get_or_create_collection(name="class_materials")

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
        
        for i, chunk in enumerate(text_chunks):
            doc_id = f"{file}_{i}"  # Unique ID for each chunk
            embedding = model.encode(chunk).tolist()
            
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{"file": file, "chunk_id": i, "text": chunk}]
            )

print("Knowledge base has been created with ChromaDB.")
