import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

pdf_text = extract_text_from_pdf("materials/pdfs/01_GEE_M.EIC_2023_2024_Intro.pdf")