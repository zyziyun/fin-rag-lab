"""
FastAPI demo server.

Two endpoints + a health check:

  POST /ingest    body: {"path": "/abs/path/to.pdf", "max_pages": int?}
                  returns: {"document_id", "n_blocks", "n_chunks", "cost_usd"}
  
  POST /query     body: {"question": "..."}
                  returns: {"answer", "citations", "stages", "query_type"}
  
  GET  /health    returns: {"status": "ok", "n_documents_indexed": int}

This is a DEMO server for the lab. Production hardening (auth, rate limit,
request validation, async streaming) is out of scope — see Module 13.

Run from repo root:
    uvicorn src.api.server:app --reload --port 8000
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.cache import CacheBundle
from src.core.config import settings, configure_langsmith
from src.observability import CostTracker
from src.pipelines import IngestionPipeline, QueryPipeline
from src.chunkers import RecursiveChunker
from src.retrievers import VectorRetriever, BM25Retriever, HybridRetriever
from src.generators import RAGGenerator


# =============================================================
# Request / Response models
# =============================================================
class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute or repo-relative path to a PDF")
    max_pages: Optional[int] = None
    page_range: Optional[tuple[int, int]] = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    n_blocks: int
    n_chunks: int
    cost_usd: float
    cache_hit: bool


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class CitationModel(BaseModel):
    chunk_id: str
    text_preview: str
    page_number: Optional[int] = None
    heading_path: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    refused: bool
    query_type: Optional[str] = None
    stages: list[str] = []
    n_chunks_retrieved: int


# =============================================================
# App state — built lazily so import doesn't require API keys
# =============================================================
class AppState:
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.cache = CacheBundle.from_root("cache", enabled=True)
        self.ingestion = IngestionPipeline(
            cache=self.cache, cost_tracker=self.cost_tracker
        )
        # Retrievers + generator are built lazily after first ingest
        self.vector: Optional[VectorRetriever] = None
        self.bm25: Optional[BM25Retriever] = None
        self.hybrid: Optional[HybridRetriever] = None
        self.generator: Optional[RAGGenerator] = None
        self.query_pipeline: Optional[QueryPipeline] = None
        self.indexed_docs: list[str] = []
    
    def ensure_retrievers(self):
        if self.vector is None:
            self.vector = VectorRetriever(
                persist_dir="./chroma_db_api",
                embeddings_cache_dir=self.cache.embeddings_dir,
            )
            self.bm25 = BM25Retriever()
            self.hybrid = HybridRetriever(self.vector, self.bm25)
            self.generator = RAGGenerator(cost_tracker=self.cost_tracker)
            self.query_pipeline = QueryPipeline(
                self.hybrid, self.generator, cost_tracker=self.cost_tracker,
            )


def build_app() -> FastAPI:
    """Factory pattern — lets tests build an isolated app."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_langsmith()
        yield
    
    app = FastAPI(
        title="VoyageAI RAG Lab API",
        version="1.0.0",
        description="Demo RAG service. POST /ingest, then POST /query.",
        lifespan=lifespan,
    )
    state = AppState()
    
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "n_documents_indexed": len(state.indexed_docs),
            "openai_key_set": settings.has_openai_key,
        }
    
    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest):
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(404, f"file not found: {path}")
        
        state.ensure_retrievers()
        
        report = state.ingestion.ingest(
            path, max_pages=req.max_pages, page_range=req.page_range,
        )
        chunks = RecursiveChunker().chunk(report.document)
        state.hybrid.index(chunks)
        state.indexed_docs.append(report.document.document_id)
        
        return IngestResponse(
            document_id=report.document.document_id,
            title=report.document.title,
            n_blocks=len(report.document.blocks),
            n_chunks=len(chunks),
            cost_usd=report.total_cost_usd,
            cache_hit=report.parse_cache_hit,
        )
    
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        state.ensure_retrievers()
        if not state.indexed_docs:
            raise HTTPException(400, "No documents indexed. POST /ingest first.")
        
        result = state.query_pipeline.query(req.question)
        chunk_lookup = {c.chunk_id: c for c in result["chunks"]}
        citations = [
            CitationModel(
                chunk_id=cid,
                text_preview=chunk_lookup[cid].text[:150] + ("..." if len(chunk_lookup[cid].text) > 150 else ""),
                page_number=chunk_lookup[cid].page_number,
                heading_path=chunk_lookup[cid].heading_path,
            )
            for cid in result["citations"] if cid in chunk_lookup
        ]
        
        return QueryResponse(
            answer=result["answer"],
            citations=citations,
            refused=result["refused"],
            query_type=result.get("query_type"),
            stages=result.get("stages", []),
            n_chunks_retrieved=len(result["chunks"]),
        )
    
    return app


# Module-level app for `uvicorn src.api.server:app`
app = build_app()
