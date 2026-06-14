from chromadb.utils import embedding_functions
from typing import List
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.dimension = 384

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [list(e) for e in self.ef(texts)]

    def embed_query(self, query: str) -> List[float]:
        return list(self.ef([query])[0])

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
