"""
main.py
-------
Command-line interface for the Document Question Answering (RAG) system.

Usage:
    # 1) Ingest documents (a single file or a whole folder)
    python main.py ingest sample_docs/

    # 2) Ask a question (uses the index built in step 1)
    python main.py query "What is the main idea of the document?"

    # 3) Or just chat interactively after ingesting
    python main.py chat sample_docs/

Generation backend is auto-selected (Claude API if ANTHROPIC_API_KEY is set,
else a local HF model, else a no-model extractive fallback). Force one with
--backend anthropic|local|extractive.
"""

import argparse
import os
from rag.pipeline import RAGPipeline

INDEX_PATH = "data/index"


def cmd_ingest(args):
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        generation_backend=args.backend,
    )
    n_chunks = pipeline.ingest(args.source)
    pipeline.save(INDEX_PATH)
    print(f"Ingested and indexed {n_chunks} chunks from '{args.source}'.")
    print(f"Embedding backend used: {pipeline.embedder.backend}")
    print(f"Index saved to '{INDEX_PATH}'.")


def cmd_query(args):
    pipeline = RAGPipeline(top_k=args.top_k, generation_backend=args.backend)
    if not os.path.exists(INDEX_PATH + ".chunks.json"):
        raise SystemExit("No index found. Run 'python main.py ingest <path>' first.")
    pipeline.load(INDEX_PATH)
    result = pipeline.query(args.question)
    _print_result(result)


def cmd_chat(args):
    """Convenience command: ingest once, then loop asking questions."""
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        generation_backend=args.backend,
    )
    n_chunks = pipeline.ingest(args.source)
    print(f"Ingested {n_chunks} chunks. Embedding backend: {pipeline.embedder.backend}")
    print("Type your question (or 'exit' to quit):\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        result = pipeline.query(question)
        _print_result(result)


def _print_result(result):
    print("\n" + "=" * 60)
    print(f"Q: {result['question']}")
    print("-" * 60)
    print(f"A: {result['answer']}")
    print("-" * 60)
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score={s['score']}): {s['preview']}...")
    print("=" * 60 + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Document Question Answering (RAG) system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load, chunk, embed, and index documents")
    p_ingest.add_argument("source", help="Path to a document file or a folder of documents")
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--chunk-overlap", type=int, default=150)
    p_ingest.add_argument("--backend", default="auto", choices=["auto", "anthropic", "local", "extractive"])
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a question against the saved index")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=4)
    p_query.add_argument("--backend", default="auto", choices=["auto", "anthropic", "local", "extractive"])
    p_query.set_defaults(func=cmd_query)

    p_chat = sub.add_parser("chat", help="Ingest once, then ask questions interactively")
    p_chat.add_argument("source")
    p_chat.add_argument("--chunk-size", type=int, default=800)
    p_chat.add_argument("--chunk-overlap", type=int, default=150)
    p_chat.add_argument("--top-k", type=int, default=4)
    p_chat.add_argument("--backend", default="auto", choices=["auto", "anthropic", "local", "extractive"])
    p_chat.set_defaults(func=cmd_chat)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
