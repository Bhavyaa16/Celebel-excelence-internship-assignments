"""
embedder.py
-----------
Stage 3 of the RAG pipeline: Embedding Creation.

Converts text into vector representations that capture semantic meaning.

Two backends are supported:
  1. "sentence-transformers" (semantic, recommended) - requires internet
     the first time to download the model, then works offline.
  2. "tfidf" (lexical fallback) - pure scikit-learn, no downloads needed.
     Used automatically if sentence-transformers / its model isn't available,
     so the pipeline always runs end-to-end.
"""

from typing import List
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", prefer_semantic: bool = True):
        self.backend = None
        self.model = None
        self._tfidf_vectorizer = None

        if prefer_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
            except Exception as e:
                print(
                    f"[info] Falling back to TF-IDF embeddings "
                    f"(sentence-transformers unavailable: {e})"
                )

        if self.backend is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            self.backend = "tfidf"

    def fit(self, texts: List[str]) -> None:
        """Required for the TF-IDF backend to build its vocabulary; no-op for
        sentence-transformers, which needs no corpus-specific fitting."""
        if self.backend == "tfidf":
            self._tfidf_vectorizer.fit(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return np.array(self.model.encode(texts, show_progress_bar=False))
        else:
            return self._tfidf_vectorizer.transform(texts).toarray()

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
