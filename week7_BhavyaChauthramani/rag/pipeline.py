"""
pipeline.py
-----------
Ties every stage together into one RAGPipeline object:

  ingest()  -> load documents, chunk, embed, store          (stages 1-4)
  query()   -> embed question, retrieve, generate answer    (stages 5-7)
"""

from typing import List, Dict
from .document_loader import load_documents
from .chunker import chunk_documents
from .embedder import Embedder
from .vector_store import VectorStore
from .generator import generate_answer


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 4,
        generation_backend: str = "auto",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.generation_backend = generation_backend

        self.embedder = Embedder()
        self.store = VectorStore()

    def ingest(self, source: str) -> int:
        """Load, chunk, embed, and index document(s) from a file or folder.
        Returns the number of chunks indexed."""
        documents = load_documents(source)
        chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        texts = [c["text"] for c in chunks]

        self.embedder.fit(texts)
        embeddings = self.embedder.encode(texts)
        self.store.add(embeddings, chunks)
        return len(chunks)

    def retrieve(self, question: str):
        query_vec = self.embedder.encode_query(question)
        return self.store.search(query_vec, top_k=self.top_k)

    def query(self, question: str) -> Dict:
        retrieved = self.retrieve(question)
        answer = generate_answer(question, retrieved, backend=self.generation_backend)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"source": c["source"], "score": round(score, 4), "preview": c["text"][:150]}
                for c, score in retrieved
            ],
        }

    def save(self, path: str = "data/index"):
        self.store.save(path)

    def load(self, path: str = "data/index"):
        self.store.load(path)
        # The TF-IDF backend needs its vocabulary rebuilt from the indexed
        # chunks on every fresh process (sentence-transformers needs no fit,
        # so this is a no-op for that backend).
        self.embedder.fit([c["text"] for c in self.store.chunks])
