"""
vector_store.py
----------------
Stage 4 of the RAG pipeline: Vector Database.

A minimal in-memory vector store using cosine similarity. Deliberately
simple (numpy only, no external service) so the whole project runs with
zero infrastructure - swap this out for FAISS/Chroma/Pinecone/Weaviate
for larger, persistent, production-scale document sets.
"""

import json
import os
from typing import List, Dict, Tuple
import numpy as np


class VectorStore:
    def __init__(self):
        self.embeddings: np.ndarray = None
        self.chunks: List[Dict[str, str]] = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, str]]) -> None:
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        norms[norms == 0] = 1e-10
        return (self.embeddings @ query_vec) / norms

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> List[Tuple[Dict[str, str], float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        scores = self._cosine_similarity(query_vec)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path + ".embeddings.npy", self.embeddings)
        with open(path + ".chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)

    def load(self, path: str) -> None:
        self.embeddings = np.load(path + ".embeddings.npy")
        with open(path + ".chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
