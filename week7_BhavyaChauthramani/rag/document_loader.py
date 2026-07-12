"""
document_loader.py
-------------------
Stage 1 of the RAG pipeline: Document Ingestion.

Loads documents (PDF, TXT, MD) from a file or a folder and converts
them into raw text, tagged with their source filename.
"""

import os
from typing import List, Dict


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required to read PDF files. Install it with: pip install pypdf"
        ) from e

    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_parts.append(extracted)
    return "\n".join(text_parts)


def load_document(path: str) -> str:
    """Load a single document (.pdf, .txt, .md) and return its raw text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext in (".txt", ".md"):
        return _load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (supported: .pdf, .txt, .md)")


def load_documents(source: str) -> List[Dict[str, str]]:
    """
    Load one document, or every supported document inside a folder.

    Returns a list of dicts: [{"source": filename, "text": raw_text}, ...]
    Empty/unreadable files are skipped with a warning instead of crashing
    the whole ingestion run.
    """
    documents = []

    if os.path.isdir(source):
        filenames = sorted(os.listdir(source))
        paths = [os.path.join(source, f) for f in filenames]
    else:
        paths = [source]

    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".pdf", ".txt", ".md"):
            continue
        try:
            text = load_document(path)
            if text.strip():
                documents.append({"source": os.path.basename(path), "text": text})
            else:
                print(f"[warn] '{path}' produced no extractable text, skipping.")
        except Exception as e:
            print(f"[warn] could not read '{path}': {e}")

    if not documents:
        raise FileNotFoundError(
            f"No readable .pdf/.txt/.md documents found at '{source}'."
        )

    return documents
