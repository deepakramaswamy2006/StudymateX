import re
from typing import List, Dict

def simple_word_tokenize(text: str):
    return re.findall(r"\S+", text)

def make_chunks(text: str, chunk_size=500, overlap=100) -> List[Dict]:
    tokens = simple_word_tokenize(text)
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(tokens):
        j = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:j]
        chunk_text = " ".join(chunk_tokens)
        chunks.append({"chunk_id": chunk_id, "text": chunk_text})
        chunk_id += 1
        if j == len(tokens):
            break
        i = j - overlap
    return chunks