import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

def extract_text_from_pdf(path: Path) -> str:
    doc = fitz.open(path)
    text = "\n\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text

def process_uploaded_files(file_paths: List[Path]) -> List[Dict]:
    corpus = []
    for p in file_paths:
        text = extract_text_from_pdf(p)
        corpus.append({"path": str(p), "text": text})
    return corpus