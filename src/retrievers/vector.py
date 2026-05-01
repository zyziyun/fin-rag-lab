"""
Vector retriever using Chroma + OpenAI embeddings.

Embedding cache: if `embeddings_cache_dir` is provided, wraps the embeddings
with LangChain's CacheBackedEmbeddings. Re-running with the same chunks
returns instantly + costs $0.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document as LCDocument
from langsmith import traceable

from src.core.interfaces import BaseRetriever
from src.core.models import DocumentChunk
from src.core.config import settings
from src.core.cache import make_cached_embeddings


class VectorRetriever(BaseRetriever):
    name = "vector"
    
    def __init__(
        self,
        persist_dir: str | Path = "./chroma_db",
        collection: str = "voyageai",
        embedding_model: Optional[str] = None,
        embeddings_cache_dir: Optional[Path | str] = None,
    ):
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        
        self.persist_dir = str(persist_dir)
        self.collection = collection
        self.embedding_model = embedding_model or settings.embedding_model
        
        base_embeddings = OpenAIEmbeddings(model=self.embedding_model)
        if embeddings_cache_dir:
            self.embeddings = make_cached_embeddings(
                base_embeddings,
                cache_dir=embeddings_cache_dir,
                namespace=self.embedding_model,
                enabled=True,
            )
        else:
            self.embeddings = base_embeddings
        
        self.store = Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
    
    def index(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        lc_docs = [
            LCDocument(
                page_content=c.text,
                metadata={
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "parent_chunk_id": c.parent_chunk_id or "",
                    "heading_path": " > ".join(c.heading_path),
                    "page_number": c.page_number or 0,
                },
            )
            for c in chunks
        ]
        ids = [c.chunk_id for c in chunks]
        self.store.add_documents(lc_docs, ids=ids)
    
    @traceable(name="vector_search")
    def retrieve(self, query: str, k: int = 5) -> list[DocumentChunk]:
        results = self.store.similarity_search_with_score(query, k=k)
        return [self._lc_to_chunk(doc) for doc, _ in results]
    
    def search_with_scores(self, query: str, k: int = 10) -> list[tuple[DocumentChunk, float]]:
        results = self.store.similarity_search_with_score(query, k=k)
        return [(self._lc_to_chunk(doc), score) for doc, score in results]
    
    def reset(self):
        try:
            self.store.delete_collection()
        except Exception:
            pass
        from langchain_chroma import Chroma
        self.store = Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
    
    @staticmethod
    def _lc_to_chunk(lc_doc: LCDocument) -> DocumentChunk:
        m = lc_doc.metadata or {}
        heading_str = m.get("heading_path", "")
        return DocumentChunk(
            chunk_id=m.get("chunk_id", "") or f"unknown_{id(lc_doc)}",
            document_id=m.get("document_id", ""),
            text=lc_doc.page_content,
            parent_chunk_id=m.get("parent_chunk_id") or None,
            heading_path=heading_str.split(" > ") if heading_str else [],
            page_number=m.get("page_number") or None,
        )
