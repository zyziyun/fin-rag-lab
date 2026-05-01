"""S4 §4.3 Strategy B: Recursive splitting that respects block boundaries.

Three production tricks vs vanilla LangChain usage:
  1. Each block is split independently → chunks never cross block boundaries
  2. Token-based length function (tiktoken)
  3. heading_path is prepended to every chunk's text → cheap context enrichment
"""
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter, HTMLHeaderTextSplitter, LatexTextSplitter, RecursiveJsonSplitter

from src.core.interfaces import BaseChunker
from src.core.models import Document, DocumentChunk
from ._token_utils import get_token_counter


class RecursiveChunker(BaseChunker):
    name = "recursive"
    
    # Block types we DON'T directly emit (they become context for paragraphs)
    _SKIP_AS_CHUNK = {"h1", "h2", "h3", "h4", "title", "header", "footer"}
    
    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 60,
        model: str = "gpt-4o",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = model
    
    def chunk(self, doc: Document) -> list[DocumentChunk]:
        token_count = get_token_counter(self.model)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=token_count,
        )
        
        chunks: list[DocumentChunk] = []
        for block in doc.blocks:
            if block.block_type in self._SKIP_AS_CHUNK:
                continue
            text = block.get_embed_text()
            if not text.strip():
                continue
            
            heading_prefix = " > ".join(block.heading_path)
            for sub_text in splitter.split_text(text):
                full = (
                    f"[Section: {heading_prefix}]\n{sub_text}"
                    if heading_prefix else sub_text
                )
                chunks.append(DocumentChunk(
                    document_id=doc.document_id,
                    text=full,
                    source_block_ids=[block.block_id],
                    heading_path=block.heading_path,
                    page_number=block.page_number,
                    metadata={"chunker": self.name},
                ))
        return chunks
