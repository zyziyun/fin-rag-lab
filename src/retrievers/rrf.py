"""
Reciprocal Rank Fusion (Cormack et al., 2009).

Score-agnostic merge: works when the input rankings have incomparable scores
(vector cosine vs BM25 raw score).

Formula:  RRF_score(d) = Σ over rankings r:  1 / (k + rank_r(d))
"""
from __future__ import annotations
from src.core.models import DocumentChunk


def rrf_merge(
    rankings: list[list[tuple[DocumentChunk, float]]],
    k: int = 60,
    top_n: int = 10,
) -> list[tuple[DocumentChunk, float]]:
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, DocumentChunk] = {}
    
    for ranking in rankings:
        for rank_idx, (chunk, _orig_score) in enumerate(ranking):
            rank = rank_idx + 1
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
    
    sorted_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)
    return [(chunk_lookup[cid], rrf_scores[cid]) for cid in sorted_ids[:top_n]]
