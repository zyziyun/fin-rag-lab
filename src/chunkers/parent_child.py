"""
S4 §4.5 Strategy D: Parent-Child (small-to-big retrieval).

Core insight:
  "The best chunk size for retrieval is not the best chunk size for generation."

Children are tiny (~150 tokens) → precise embedding match.
Parents are big (~800 tokens) → enough context for the LLM to reason.
Each child has parent_chunk_id pointing back to its parent.
"""
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces import BaseChunker
from src.core.models import Document, DocumentChunk
from ._token_utils import get_token_counter


class ParentChildChunker(BaseChunker):
    name = "parent_child"
    
    def __init__(
        self,
        parent_size: int = 800,
        child_size: int = 150,
        parent_overlap: int = 80,
        child_overlap: int = 20,
        model: str = "gpt-4o",
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.model = model
    
    def chunk(self, doc: Document) -> list[DocumentChunk]:
        """Returns children only (what goes in vector DB)."""
        _, children = self.chunk_with_parents(doc)
        return children
    
    def chunk_with_parents(
        self, doc: Document
    ) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
        token_count = get_token_counter(self.model)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=self.parent_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=token_count,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=token_count,
        )
        
        # Build full doc text (with heading prefixes inline) AND track block char positions
        parts: list[str] = []
        # Each entry: (char_start, char_end, page_number)
        block_pos: list[tuple[int, int, int | None]] = []
        cursor = 0
        for block in doc.blocks:
            if block.block_type in ("header", "footer"):
                continue
            prefix = " > ".join(block.heading_path)
            text = block.get_embed_text()
            if not text:
                continue
            if block.block_type in ("h1", "h2", "h3"):
                segment = f"\n## {text}\n"
            else:
                segment = f"[{prefix}]\n{text}\n\n" if prefix else f"{text}\n\n"
            parts.append(segment)
            block_pos.append((cursor, cursor + len(segment), block.page_number))
            cursor += len(segment)
        full_text = "".join(parts)
        
        def _page_for(chunk_text: str, search_from: int = 0) -> tuple[int | None, int]:
            """Locate chunk in full_text and return (page_number, end_pos) for next search."""
            idx = full_text.find(chunk_text, search_from)
            if idx < 0:
                return (None, search_from)
            chunk_end = idx + len(chunk_text)
            for s, e, pg in block_pos:
                if s < chunk_end and e > idx and pg is not None:
                    return (pg, chunk_end)
            return (None, chunk_end)
        
        parents: list[DocumentChunk] = []
        children: list[DocumentChunk] = []
        parent_search_pos = 0
        
        for parent_text in parent_splitter.split_text(full_text):
            parent_page, parent_search_pos = _page_for(parent_text, parent_search_pos)
            parent = DocumentChunk(
                document_id=doc.document_id,
                text=parent_text,
                page_number=parent_page,
                metadata={"chunker": self.name, "level": "parent"},
            )
            parents.append(parent)
            for child_text in child_splitter.split_text(parent_text):
                # Children inherit parent's page (close enough — they're contiguous within parent)
                children.append(DocumentChunk(
                    document_id=doc.document_id,
                    text=child_text,
                    parent_chunk_id=parent.chunk_id,
                    page_number=parent_page,
                    metadata={"chunker": self.name, "level": "child"},
                ))
        
        return parents, children
