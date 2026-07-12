"""
generator.py
------------
Stage 7 of the RAG pipeline: Answer Generation.

Given a question + retrieved context chunks, produce a grounded answer.

Three backends, chosen automatically (best available) or explicitly:
  - "anthropic"  : Claude API (needs ANTHROPIC_API_KEY env var). Best quality.
  - "local"      : a small local HF seq2seq model (google/flan-t5-base).
                   Needs `transformers` + `torch` and a one-time model download.
  - "extractive" : no model at all - returns the most relevant sentences
                   from the retrieved chunks. Always works, zero setup.
                   Good fallback / offline demo mode.
"""

import os
import re
from typing import List, Dict


PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know - do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def _build_context(retrieved: List[Dict[str, str]]) -> str:
    parts = []
    for chunk, score in retrieved:
        parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _extractive_answer(question: str, retrieved: List[Dict[str, str]]) -> str:
    """No-LLM fallback: surface the most relevant sentence(s) as the answer."""
    if not retrieved:
        return "I couldn't find anything relevant in the documents to answer that."

    q_words = set(re.findall(r"\w+", question.lower()))
    best_sentence, best_overlap, best_source = None, -1, None

    for chunk, _score in retrieved:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
        for sentence in sentences:
            s_words = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(q_words & s_words)
            if overlap > best_overlap and len(sentence.strip()) > 15:
                best_overlap, best_sentence, best_source = overlap, sentence.strip(), chunk["source"]

    top_chunk_text = retrieved[0][0]["text"].strip()
    if best_sentence:
        return f"{best_sentence}\n\n(Extracted from: {best_source})"
    return f"{top_chunk_text[:400]}...\n\n(Extracted from: {retrieved[0][0]['source']})"


def _anthropic_answer(question: str, retrieved: List[Dict[str, str]]) -> str:
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    context = _build_context(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


_local_pipeline = None


def _local_answer(question: str, retrieved: List[Dict[str, str]]) -> str:
    global _local_pipeline
    if _local_pipeline is None:
        from transformers import pipeline
        _local_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    context = _build_context(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = _local_pipeline(prompt, max_length=256, do_sample=False)
    return result[0]["generated_text"].strip()


def generate_answer(question: str, retrieved: List[Dict[str, str]], backend: str = "auto") -> str:
    """
    Generate an answer grounded in `retrieved` chunks.

    backend: "auto" (try anthropic -> local -> extractive), or force one of
             "anthropic", "local", "extractive".
    """
    if backend == "extractive":
        return _extractive_answer(question, retrieved)

    if backend in ("anthropic", "auto") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_answer(question, retrieved)
        except Exception as e:
            if backend == "anthropic":
                raise
            print(f"[info] Anthropic backend failed ({e}), trying local model...")

    if backend in ("local", "auto"):
        try:
            return _local_answer(question, retrieved)
        except Exception as e:
            if backend == "local":
                raise
            print(f"[info] Local model backend unavailable ({e}), using extractive fallback...")

    return _extractive_answer(question, retrieved)
