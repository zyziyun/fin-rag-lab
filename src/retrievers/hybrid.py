"""
HybridRetriever: composes Vector + BM25 + RRF + (optional) parent-child swap.

This is what production RAG looks like in 2026:
  1. Run vector and BM25 in parallel (both retrieve top-fetch_k)
  2. Merge with RRF (k=60) into a single ranked list
  3. If parent-child enabled, swap each child for its parent (deduped)
  4. Return top-k
"""
from __future__ import annotations
from typing import Optional

from langsmith import traceable

from src.core.interfaces import BaseRetriever
from src.core.models import DocumentChunk
from .rrf import rrf_merge


class HybridRetriever(BaseRetriever):
    name = "hybrid"
    
    def __init__(
        self,
        vector,                         # VectorRetriever
        bm25,                           # BM25Retriever
        parent_store: Optional[dict[str, DocumentChunk]] = None,
        rrf_k: int = 60,
    ):
        self.vector = vector
        self.bm25 = bm25
        self.parent_store = parent_store or {}
        self.rrf_k = rrf_k
    
    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index into both vector + BM25."""
        self.vector.index(chunks)
        self.bm25.index(chunks)
    
    @traceable(name="hybrid_retrieve")
    def retrieve(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        use_parent: bool = True,
    ) -> list[DocumentChunk]:
        vec_scored = self.vector.search_with_scores(query, k=fetch_k)
        bm25_scored = self.bm25.search_with_scores(query, k=fetch_k)
        
        fused = rrf_merge([vec_scored, bm25_scored], k=self.rrf_k, top_n=fetch_k)
        
        if use_parent and self.parent_store:
            return self._swap_to_parents(fused, k)
        return [c for c, _ in fused[:k]]
    
    def _swap_to_parents(
        self, fused: list[tuple[DocumentChunk, float]], k: int
    ) -> list[DocumentChunk]:
        seen: set[str] = set()
        out: list[DocumentChunk] = []
        for child, _ in fused:
            pid = child.parent_chunk_id
            if pid:
                if pid in seen or pid not in self.parent_store:
                    continue
                seen.add(pid)
                out.append(self.parent_store[pid])
            else:
                out.append(child)
            if len(out) >= k:
                break
        return out
