"""
Coverage diagnostics — corpus-agnostic tools for measuring retrieval quality.

This is the "B" approach from the Phase 2 design discussion: instead of
hard-coding the WF-misses-tables observation into 02_chunking, we build
**reusable tools** that quantify retrieval quality on any corpus.

Two tools:

  1. CoverageDiagnostic.diagnose(retriever, queries) → per-query DataFrame
     showing what kinds of blocks were retrieved and a simple "numeric content"
     proxy for "did we get the table data?"
  
  2. compare_strategies(chunkers, doc, queries) → comparison DataFrame across
     chunking strategies — same retriever applied to each chunker's output.

These are diagnostic, not absolute metrics. For ground-truth scoring, use
RagasEvaluator on a curated golden set.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.interfaces import BaseRetriever, BaseChunker
from src.core.models import Document, DocumentChunk


# Regex matching a single "number-ish" token (currency, percent, decimal, plain digits).
# Allows trailing punctuation (commas, periods at end of words) and unit suffixes.
_NUMBER_TOKEN_RE = re.compile(
    r"^\$?\d[\d,]*(?:\.\d+)?(?:%|B|bn|M|mn|k)?[.,;:)\]]?$",
    re.IGNORECASE,
)


def _numeric_density(text: str) -> float:
    """Fraction of whitespace-separated tokens that look like numbers.
    
    Discriminates "table-ish" content (high density, e.g., '164.7 160.7 139.1' → 1.0)
    from "prose with a few numbers" (low density, e.g., dates and section numbers → ~0.05).
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _NUMBER_TOKEN_RE.match(t)) / len(tokens)


def _is_data_dense(text: str, threshold: float = 0.20) -> bool:
    """True iff numeric_density(text) >= threshold.
    
    threshold=0.20 empirically separates table-fragment chunks (≥0.30 typical)
    from prose chunks with date/dollar mentions (≤0.10 typical) on financial earnings PDFs.
    Short text fragments may give noisy results; this is designed for chunk-sized text (200+ tokens).
    """
    return _numeric_density(text) >= threshold


@dataclass
class QueryDiagnostic:
    query: str
    n_retrieved: int
    pct_dense: float           # fraction of chunks where >= 15% of tokens are number-like
    avg_numeric_density: float         # average fraction of number-tokens across retrieved chunks
    n_unique_pages: int               # how many distinct pages contributed
    avg_chunk_chars: float
    snippet: str                       # text of top result, for inspection


class CoverageDiagnostic:
    """
    Diagnose what a retriever actually returns for a list of queries.
    
    Use this to answer: 'For my chunking strategy + retriever, do queries about
    numerical facts retrieve chunks that contain numbers, or just prose?'
    """
    
    def __init__(self, retriever: BaseRetriever, k: int = 5):
        self.retriever = retriever
        self.k = k
    
    def diagnose(self, queries: list[str]) -> list[QueryDiagnostic]:
        out = []
        for q in queries:
            chunks = self.retriever.retrieve(q, k=self.k)
            if not chunks:
                out.append(QueryDiagnostic(
                    query=q, n_retrieved=0, pct_dense=0.0, avg_numeric_density=0.0,
                    n_unique_pages=0, avg_chunk_chars=0.0, snippet="(no results)",
                ))
                continue
            n_with_nums = sum(1 for c in chunks if _is_data_dense(c.text))
            avg_density = sum(_numeric_density(c.text) for c in chunks) / len(chunks)
            pages = {c.page_number for c in chunks if c.page_number is not None}
            avg_chars = sum(len(c.text) for c in chunks) / len(chunks)
            top_text = chunks[0].text[:140].replace("\n", " ")
            out.append(QueryDiagnostic(
                query=q,
                n_retrieved=len(chunks),
                pct_dense=n_with_nums / len(chunks),
                avg_numeric_density=avg_density,
                n_unique_pages=len(pages),
                avg_chunk_chars=avg_chars,
                snippet=top_text,
            ))
        return out
    
    @staticmethod
    def to_dataframe(diagnostics: list[QueryDiagnostic]):
        import pandas as pd
        return pd.DataFrame([
            {
                "query": d.query,
                "n_retrieved": d.n_retrieved,
                "pct_dense": round(d.pct_dense, 2),
                "avg_density": round(d.avg_numeric_density, 3),
                "n_unique_pages": d.n_unique_pages,
                "avg_chars": round(d.avg_chunk_chars, 0),
                "top_snippet": d.snippet,
            }
            for d in diagnostics
        ])


def compare_strategies(
    chunkers: dict[str, BaseChunker],
    doc: Document,
    queries: list[str],
    retriever_factory,
    k: int = 5,
):
    """
    Compare multiple chunking strategies on the same document + same queries.
    
    Args:
        chunkers: {strategy_name: chunker_instance}
        doc: Document to chunk (output of IngestionPipeline.ingest)
        queries: queries to evaluate
        retriever_factory: callable() → fresh BaseRetriever (must support .index() and
                           .search_with_scores() / .retrieve()). Called once per strategy
                           to create an isolated retriever.
        k: top-k retrieved per query
    
    Returns: pandas.DataFrame indexed by (strategy, query) with diagnostic columns
    """
    import pandas as pd
    rows = []
    for strategy_name, chunker in chunkers.items():
        chunks = chunker.chunk(doc)
        retriever = retriever_factory()
        retriever.index(chunks)
        diag = CoverageDiagnostic(retriever, k=k).diagnose(queries)
        for d in diag:
            rows.append({
                "strategy": strategy_name,
                "n_chunks_total": len(chunks),
                "query": d.query,
                "n_retrieved": d.n_retrieved,
                "pct_dense": round(d.pct_dense, 2),
                "avg_density": round(d.avg_numeric_density, 3),
                "n_unique_pages": d.n_unique_pages,
                "avg_chars": round(d.avg_chunk_chars, 0),
            })
    return pd.DataFrame(rows)
