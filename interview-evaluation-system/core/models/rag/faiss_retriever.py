import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.interfaces.retriever import RetrieverInterface

class FAISSRetriever(RetrieverInterface):
    """
    Dense retriever using FAISS + Sentence Transformers.
    """

    def __init__(self, index_path: str, corpus: list):
        self.index = faiss.read_index(index_path)
        self.corpus = corpus
        self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        if not query:
            return []

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)

        _, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.corpus[idx])

        return results
