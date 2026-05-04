<<<<<<< HEAD
"""Module 3: Reranking - Cross-encoder top-20 -> top-3 + latency benchmark."""

import os
import re
import sys
import time
=======
"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
>>>>>>> origin/main
from dataclasses import dataclass

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
<<<<<<< HEAD
        self._model_kind = None
        self._model_load_attempted = False
        # Set USE_REAL_RERANKER=1 to force real model loading.
        self.use_real_model = os.getenv("USE_REAL_RERANKER", "0") == "1"

    def _load_model(self):
        if self._model_load_attempted:
            return self._model

        self._model_load_attempted = True
        if not self.use_real_model:
            return None

        try:
            import torch
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_name, use_fp16=torch.cuda.is_available())
            self._model_kind = "flag"
            return self._model
        except Exception:
            self._model = None
            self._model_kind = None

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._model_kind = "cross"
            return self._model
        except Exception:
            self._model = None
            self._model_kind = None
            return None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _lexical_score(self, query: str, document: str, original_score: float) -> float:
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(document)
        if not q_tokens or not d_tokens:
            return original_score * 0.01

        q_set = set(q_tokens)
        d_set = set(d_tokens)
        overlap = len(q_set.intersection(d_set))
        coverage = overlap / max(len(q_set), 1)
        phrase_bonus = 1.0 if query.lower() in document.lower() else 0.0

        q_numbers = re.findall(r"\d+", query)
        d_numbers = re.findall(r"\d+", document)
        number_bonus = sum(1 for n in q_numbers if n in d_numbers)

        return (
            coverage * 5.0
            + overlap * 0.3
            + phrase_bonus * 0.5
            + number_bonus * 0.5
            + original_score * 0.1
        )

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 -> top-k."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        pairs = [(query, str(doc.get("text", ""))) for doc in documents]
        scores = []

        if model is not None:
            try:
                if self._model_kind == "flag":
                    raw_scores = model.compute_score(pairs)
                elif self._model_kind == "cross":
                    raw_scores = model.predict(pairs)
                else:
                    raw_scores = []
                scores = [float(s) for s in raw_scores]
            except Exception:
                scores = []

        if len(scores) != len(documents):
            scores = [
                self._lexical_score(query, str(doc.get("text", "")), float(doc.get("score", 0.0)))
                for doc in documents
            ]

        combined = list(zip(scores, documents))
        combined.sort(key=lambda x: x[0], reverse=True)

        out = []
        for idx, (score, doc) in enumerate(combined[:top_k], start=1):
            out.append(
                RerankResult(
                    text=str(doc.get("text", "")),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}) or {},
                    rank=idx,
                )
            )
        return out
=======

    def _load_model(self):
        if self._model is None:
            # TODO: Load cross-encoder model
            # Option A: from FlagEmbedding import FlagReranker
            #           self._model = FlagReranker(self.model_name, use_fp16=True)
            # Option B: from sentence_transformers import CrossEncoder
            #           self._model = CrossEncoder(self.model_name)
            pass
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        # TODO: Implement reranking
        # 1. model = self._load_model()
        # 2. pairs = [(query, doc["text"]) for doc in documents]
        # 3. scores = model.compute_score(pairs)  # FlagReranker
        #    OR scores = model.predict(pairs)      # CrossEncoder
        # 4. Combine: [(score, doc) for score, doc in zip(scores, documents)]
        # 5. Sort by score descending
        # 6. Return top_k RerankResult(text=..., original_score=doc["score"],
        #                              rerank_score=score, metadata=doc["metadata"], rank=i)
        return []
>>>>>>> origin/main


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
<<<<<<< HEAD

=======
>>>>>>> origin/main
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
<<<<<<< HEAD
        # Optional path. Keep graceful behavior if flashrank is not installed.
        _ = query
        _ = documents
        _ = top_k
=======
        # TODO (optional): from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
>>>>>>> origin/main
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
<<<<<<< HEAD
    if n_runs <= 0:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000.0)

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
=======
    # TODO: Implement benchmark
    # 1. times = []
    # 2. for _ in range(n_runs):
    #      start = time.perf_counter()
    #      reranker.rerank(query, documents)
    #      times.append((time.perf_counter() - start) * 1000)  # ms
    # 3. return {"avg_ms": mean(times), "min_ms": min(times), "max_ms": max(times)}
    return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
>>>>>>> origin/main
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
