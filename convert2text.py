import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

pdf_text = extract_text_from_pdf("pdfs/01_GEE_M.EIC_2023_2024_Intro.pdf")


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert extracted text into vector embeddings
texts = pdf_text.split("\n")  # Split into smaller sections
embeddings = model.encode(texts)

# Store in FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


def search_text(query):

    # Search for relevant content
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)

    return [texts[i] for i in I[0]]  # Return top matches



# Search for relevant content
#query = "Bussiness Model"
#print(search_text(query))  # Return top matches
