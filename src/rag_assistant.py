"""
Lightweight RAG layer for the dashboard's Q&A page.

Retrieval: each store-item series' backtest result (WAPE, RMSE, recent
forecast trend) is turned into a short text summary. A TF-IDF retriever
pulls the summaries most relevant to the user's question — no heavy
embedding model needed for a few hundred short documents like this.

Generation: the retrieved summaries are passed as context to a Groq-hosted
LLM, which answers strictly from that context rather than guessing.

Requires a free Groq API key, set as the GROQ_API_KEY environment variable
(see .env.example).
"""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
TOP_K = 6


def _trend_description(actual: list, forecast: list) -> str:
    change = forecast[-1] - forecast[0]
    if change > 0.1 * max(forecast[0], 1):
        return "trending up"
    if change < -0.1 * max(forecast[0], 1):
        return "trending down"
    return "roughly stable"


def _build_documents(metrics: dict) -> list[str]:
    docs = [
        f"OVERALL MODEL PERFORMANCE: WAPE={metrics['overall']['wape']:.3f}, "
        f"RMSE={metrics['overall']['rmse']:.2f} across all store-item series."
    ]
    for s in metrics["per_series"]:
        trend = _trend_description(s["actual"], s["forecast"])
        docs.append(
            f"Store {s['store']}, Item {s['item']}: forecast is {trend}. "
            f"Backtested WAPE={s['wape']:.3f}, RMSE={s['rmse']:.2f}. "
            f"Next-period forecasted demand values: {[round(v, 1) for v in s['forecast'][:14]]} "
            f"(first 14 days shown)."
        )
    return docs


class ForecastRAG:
    def __init__(self):
        with open(ARTIFACTS_DIR / "metrics.json") as f:
            self.metrics = json.load(f)

        self.documents = _build_documents(self.metrics)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set — see .env.example")
        self.client = Groq(api_key=api_key)

    def _retrieve(self, question: str, k: int = TOP_K) -> list[str]:
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_idx = np.argsort(sims)[-k:][::-1]
        return [self.documents[i] for i in top_idx]

    def answer(self, question: str) -> str:
        context_docs = self._retrieve(question)
        context = "\n".join(f"- {d}" for d in context_docs)

        prompt = (
            "You are a retail demand-forecasting assistant. Answer the "
            "question using ONLY the context below. If the context doesn't "
            "contain the answer, say so plainly instead of guessing.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        )

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content
