from embedding.embed_index import EmbeddingIndex

class Retriever:
    def __init__(self, embed_index: EmbeddingIndex):
        self.embed_index = embed_index

    def retrieve(self, query: str, top_k=3):
        q_emb = self.embed_index.embed_texts([query])
        return self.embed_index.search(q_emb, top_k=top_k)