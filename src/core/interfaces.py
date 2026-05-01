"""
Abstract interfaces for every pipeline stage.

Design choice (decision 1, option B): we define our OWN domain-specific
abstract base classes (BaseChunker, BaseRetriever, etc) so that the architecture
isn't married to LangChain's evolving API. BUT the key dataflow nodes
(Chunker, Retriever, Generator) ALSO inherit from langchain_core.runnables.Runnable
so that they can participate in LCEL pipes:

    chain = chunker | retriever | generator

This dual inheritance gives us LangChain ecosystem citizenship + clean domain types.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from langchain_core.runnables import Runnable, RunnableConfig

from .models import Document, DocumentBlock, DocumentChunk


# =============================================================
# Stage 1: Loader — bytes → raw text + structural metadata
# =============================================================
class BaseLoader(ABC):
    """Loads a file into raw text + page info. No semantic interpretation."""
    
    @abstractmethod
    def load(
        self,
        source: str | Path,
        max_pages: Optional[int] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, Any]:
        """Returns: {"pages": [{"text": str, "page_num": int, ...}], "metadata": {...}}"""
        ...


# =============================================================
# Stage 2: Parser — raw page data → structured blocks
# =============================================================
class BaseParser(ABC):
    """Detects blocks (text/table/image/heading) within loaded pages."""
    
    @abstractmethod
    def parse(self, loaded: dict[str, Any], source_path: str | Path) -> Document:
        """Returns a Document with blocks, but blocks may not yet have semantic_content."""
        ...


# =============================================================
# Stage 3: Captioner — VLM/LLM enrichment of non-text blocks (S4 §3.3)
# =============================================================
class BaseCaptioner(ABC):
    """Adds semantic_content to table/image/chart blocks via VLM/LLM."""
    
    @abstractmethod
    def caption(self, block: DocumentBlock, doc_context: str = "") -> DocumentBlock:
        """Returns the (mutated) block with semantic_content set."""
        ...
    
    def caption_all(self, doc: Document) -> Document:
        """Default impl: caption every applicable block, return doc."""
        ctx = f"This document is: {doc.title}"
        for block in doc.blocks:
            if block.block_type in ("table", "image", "chart", "figure"):
                if not block.semantic_content:    # respect cache
                    self.caption(block, doc_context=ctx)
        return doc


# =============================================================
# Stage 4: Chunker — Document → list of Chunks (S4 §4)
# =============================================================
class BaseChunker(Runnable[Document, list[DocumentChunk]], ABC):
    """Cuts a Document into Chunks. Must also yield parent chunks for parent-child."""
    
    name: str = "base"
    
    @abstractmethod
    def chunk(self, doc: Document) -> list[DocumentChunk]:
        """Returns chunks suitable for indexing (children for parent-child strategies)."""
        ...
    
    def chunk_with_parents(self, doc: Document) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
        """For non parent-child strategies, parents == [] and children == chunk(doc)."""
        return [], self.chunk(doc)
    
    # Runnable interface — lets you do `chunker.invoke(doc)` and `chunker | next_step`
    def invoke(
        self, input: Document, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[DocumentChunk]:
        return self.chunk(input)


# =============================================================
# Stage 5: Retriever — query → relevant Chunks
# =============================================================
class BaseRetriever(Runnable[str, list[DocumentChunk]], ABC):
    """Retrieves Chunks given a natural-language query."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[DocumentChunk]:
        ...
    
    @abstractmethod
    def index(self, chunks: list[DocumentChunk]) -> None:
        ...
    
    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[DocumentChunk]:
        k = (kwargs.get("k") or 5)
        return self.retrieve(input, k=k)


# =============================================================
# Stage 6: Generator — query + retrieved chunks → answer
# =============================================================
class BaseGenerator(Runnable[dict, dict], ABC):
    """Generates a grounded answer with citations."""
    
    @abstractmethod
    def generate(self, query: str, chunks: list[DocumentChunk]) -> dict[str, Any]:
        """Returns: {"answer": str, "citations": list[str]}"""
        ...
    
    def invoke(
        self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self.generate(input["query"], input["chunks"])
