import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class EmbeddingIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2", normalize=True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.index = None
        self.metadatas = []

    def embed_texts(self, texts: List[str], batch_size=32):
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        if self.normalize:
            faiss.normalize_L2(embs)
        return embs

    def build_index(self, embeddings: np.ndarray, metadatas: List[Dict]):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadatas = metadatas

    def search(self, query_emb: np.ndarray, top_k=3):
        if self.normalize:
            faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({"score": float(score), "meta": self.metadatas[idx]})
        return results