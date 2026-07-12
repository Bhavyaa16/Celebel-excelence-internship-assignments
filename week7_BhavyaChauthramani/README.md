# Document Question Answering System (RAG)

A simple, from-scratch Retrieval-Augmented Generation (RAG) pipeline that
answers questions from your own documents (PDF / TXT / MD) instead of relying
on a language model's memorized knowledge.

## How it works (pipeline stages)

```
 PDFs/TXT  →  chunk text  →  embed chunks  →  store vectors
                                                     │
 question  →  embed question  →  retrieve top-k  ────┘
                                        │
                                        ▼
                          build prompt with context
                                        │
                                        ▼
                            LLM generates the answer
```

| Stage | File | What it does |
|---|---|---|
| 1. Document Ingestion | `rag/document_loader.py` | Loads `.pdf`/`.txt`/`.md` files into raw text |
| 2. Text Chunking | `rag/chunker.py` | Splits text into overlapping ~800-char chunks |
| 3. Embedding | `rag/embedder.py` | Converts chunks to vectors |
| 4. Vector Store | `rag/vector_store.py` | In-memory cosine-similarity search |
| 5-6. Query + Retrieval | `rag/pipeline.py` | Embeds the question, fetches top-k relevant chunks |
| 7. Answer Generation | `rag/generator.py` | LLM generates an answer grounded in the retrieved context |

## Setup

```bash
pip install -r requirements.txt
```

Minimum install (`numpy`, `scikit-learn`, `pypdf`) is enough to run the whole
pipeline offline using a TF-IDF embedder and an extractive (no-LLM) answerer.
Install the optional packages for better quality:

- `sentence-transformers` → semantic embeddings (recommended)
- `transformers` + `torch` → local LLM generation (`google/flan-t5-base`)
- `anthropic` + `export ANTHROPIC_API_KEY=...` → best-quality generation via Claude

The generation backend is auto-selected in this priority order:
**Claude API → local HF model → extractive fallback**, so the project always
runs even with zero API keys or extra installs.

## Usage

```bash
# 1. Ingest your document(s) - a single file or a whole folder
python main.py ingest sample_docs/sample.txt
python main.py ingest path/to/your/notes_folder/

# 2. Ask a question
python main.py query "What is the main idea of the document?"

# 3. Or go interactive (ingest once, ask many questions)
python main.py chat sample_docs/sample.txt
```

Force a specific generation backend with `--backend anthropic|local|extractive`.

### Example

```bash
$ python main.py ingest sample_docs/sample.txt
Ingested and indexed 3 chunks from 'sample_docs/sample.txt'.

$ python main.py query "What are the four core stages of a RAG pipeline?"
Q: What are the four core stages of a RAG pipeline?
A: Document ingestion, text chunking, embedding creation, and storing
   embeddings in a vector database for similarity search.
Sources used:
  - sample.txt (score=0.81): A typical RAG pipeline has four core stages...
```

## Using your own documents

Drop any `.pdf`, `.txt`, or `.md` files into a folder (resume, class notes,
research papers, a book chapter, etc.) and run:

```bash
python main.py ingest my_docs/
python main.py query "Summarize section 2"
```

## Project structure

```
rag_qa_system/
├── main.py                  # CLI entry point
├── requirements.txt
├── sample_docs/sample.txt   # sample document to try immediately
├── data/                    # saved vector index (created after ingest)
└── rag/
    ├── document_loader.py   # stage 1
    ├── chunker.py           # stage 2
    ├── embedder.py          # stage 3
    ├── vector_store.py      # stage 4
    ├── generator.py         # stage 7
    └── pipeline.py          # wires everything together
```

## Possible improvements (see assignment brief)

- Hybrid search: combine this cosine-similarity search with keyword/BM25 search
- Add a re-ranking step (e.g. a cross-encoder) over the top-k retrieved chunks
- Swap the in-memory store for FAISS/Chroma for larger document collections
- Try different chunk sizes/overlaps and embedding models and compare answer quality
