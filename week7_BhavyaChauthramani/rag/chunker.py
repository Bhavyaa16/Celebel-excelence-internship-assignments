"""
chunker.py
----------
Stage 2 of the RAG pipeline: Text Chunking.

Splits raw document text into smaller overlapping chunks so retrieval
can find precise, relevant passages instead of whole documents.
"""

import re
from typing import List, Dict


def _split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # Basic sentence splitter (good enough for a beginner RAG project;
    # swap in nltk/spacy for production-grade sentence boundaries).
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict[str, str]]:
    """
    Split `text` into chunks of roughly `chunk_size` characters, each
    overlapping the previous one by `chunk_overlap` characters so that
    context isn't lost at chunk boundaries.

    Returns a list of dicts: {"source": ..., "chunk_id": ..., "text": ...}
    """
    sentences = _split_into_sentences(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying over the overlap tail of the previous one
            overlap_text = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    return [
        {"source": source, "chunk_id": f"{source}::chunk_{i}", "text": chunk}
        for i, chunk in enumerate(chunks)
    ]


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict[str, str]]:
    """Chunk every document in `documents` (see load_documents)."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(
            chunk_text(doc["text"], doc["source"], chunk_size, chunk_overlap)
        )
    return all_chunks
