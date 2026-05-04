"""Module 3: Reranking - Cross-encoder top-20 to top-3 plus latency benchmark."""

import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if os.getenv("USE_REAL_RERANKER") != "1":
                return None
            import torch
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, model_kwargs={"torch_dtype": torch.float16})
        return self._model

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        query_terms = Counter(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
        text_terms = Counter(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
        if not query_terms or not text_terms:
            return 0.0
        dot = sum(query_terms[token] * text_terms[token] for token in set(query_terms) & set(text_terms))
        query_norm = sqrt(sum(value * value for value in query_terms.values()))
        text_norm = sqrt(sum(value * value for value in text_terms.values()))
        return dot / (query_norm * text_norm) if query_norm and text_norm else 0.0

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents with a real model when enabled, otherwise offline lexical scoring."""
        if not documents:
            return []

        model = self._load_model()
        if model:
            pairs = [[query, doc.get("text", "")] for doc in documents]
            scores = model.predict(pairs)
        else:
            scores = [self._lexical_score(query, doc.get("text", "")) for doc in documents]

        combined = [(float(score), doc) for score, doc in zip(scores, documents)]
        combined.sort(key=lambda item: item[0], reverse=True)

        results = []
        for i, (score, doc) in enumerate(combined[:top_k]):
            results.append(
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=score,
                    metadata=doc.get("metadata", {}),
                    rank=i + 1,
                )
            )
        return results


class FlashrankReranker:
    """Lightweight optional alternative."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"id": i, "text": d.get("text", ""), "meta": d.get("metadata", {})} for i, d in enumerate(documents)]
            ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
            results = []
            for i, item in enumerate(ranked[:top_k]):
                doc_id = item["id"]
                results.append(
                    RerankResult(
                        text=item["text"],
                        original_score=documents[doc_id].get("score", 0.0),
                        rerank_score=float(item["score"]),
                        metadata=item.get("meta", {}),
                        rank=i + 1,
                    )
                )
            return results
        except Exception:
            return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    if not documents:
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for result in reranker.rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
