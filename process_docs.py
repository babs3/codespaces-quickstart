import os
import fitz  # PyMuPDF

PDF_FOLDER = "materials/pdfs"

def extract_text_from_pdf():
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"

            name_without_ext = os.path.splitext(file)[0]  # Remove the extension
            
            # Define the full file path
            file_path = os.path.join("docs", name_without_ext + ".txt")

            # Write text to the file
            with open(file_path, "w") as file:
                file.write(text)

            print(f"File created at: {file_path}")

extract_text_from_pdf()