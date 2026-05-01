"""BM25 keyword retriever using rank-bm25.

Production note: for >1M chunks, use Elasticsearch / OpenSearch instead.
For the lab (a few hundred chunks), in-memory BM25 is fine and zero-cost.
"""
from __future__ import annotations
import re
from typing import Optional

from rank_bm25 import BM25Okapi
from langsmith import traceable

from src.core.interfaces import BaseRetriever
from src.core.models import DocumentChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever(BaseRetriever):
    name = "bm25"
    
    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self._bm25: Optional[BM25Okapi] = None
    
    def index(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        if not chunks:
            self._bm25 = None
            return
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
    
    @traceable(name="bm25_search")
    def retrieve(self, query: str, k: int = 5) -> list[DocumentChunk]:
        if self._bm25 is None or not self.chunks:
            return []
        scored = self.search_with_scores(query, k=k)
        return [c for c, _ in scored]
    
    def search_with_scores(
        self, query: str, k: int = 10
    ) -> list[tuple[DocumentChunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
