"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = segment_vietnamese(text).split()
            self.corpus_tokens.append(tokens)
        
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.bm25:
            return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"torch_dtype": torch.float16})
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            return
            
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        # Encode and upsert in batches to avoid OOM on low-RAM machines
        BATCH = 4
        encoder = self._get_encoder()
        idx = 0
        for start in range(0, len(chunks), BATCH):
            batch_chunks = chunks[start:start + BATCH]
            batch_texts = [c["text"] for c in batch_chunks]
            batch_vectors = encoder.encode(batch_texts, show_progress_bar=False)
            
            points = []
            for chunk, vector in zip(batch_chunks, batch_vectors):
                payload = {**chunk.get("metadata", {}), "text": chunk["text"]}
                points.append(PointStruct(id=idx, vector=vector.tolist(), payload=payload))
                idx += 1
            self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors (qdrant-client 1.17+ uses query_points)."""
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k
            )
            
            results = []
            for point in response.points:
                results.append(SearchResult(
                    text=point.payload.get("text", ""),
                    score=float(point.score),
                    metadata={k: v for k, v in point.payload.items() if k != "text"},
                    method="dense"
                ))
            return results
        except Exception as e:
            print(f"[DenseSearch] Search error: {e}")
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            text = result.text
            if text not in rrf_scores:
                rrf_scores[text] = {"score": 0.0, "result": result}
            rrf_scores[text]["score"] += 1.0 / (k + rank + 1)
            
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    final_results = []
    for item in sorted_items:
        res = item["result"]
        final_results.append(SearchResult(
            text=res.text,
            score=item["score"],
            metadata=res.metadata,
            method="hybrid"
        ))
    return final_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")