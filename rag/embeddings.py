from abc import ABC, abstractmethod
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass

class SBERTEmbedder(Embedder):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_numpy=True))

class VectorStore(ABC):
    @abstractmethod
    def add(self, embeddings: np.ndarray, documents: List[str]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        pass

class FAISSVectorStore(VectorStore):
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.metadatas = [] 

    def add(self, embeddings: np.ndarray, documents: List[str], metadata: dict = None) -> None:
        self.index.add(embeddings)
        self.documents.extend(documents)
        if metadata:
            self.metadatas.extend([metadata] * len(documents)) 
        else:
            self.metadatas.extend([{}] * len(documents))

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

