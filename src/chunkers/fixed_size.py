"""S4 §4.2 Strategy A: Fixed-size chunking with overlap. Class-based."""
from __future__ import annotations
from src.core.interfaces import BaseChunker
from src.core.models import Document, DocumentChunk
from ._token_utils import get_encoding


class FixedSizeChunker(BaseChunker):
    """Naive token-based chunking. Ignores document structure entirely."""
    
    name = "fixed_size"
    
    def __init__(self, size: int = 500, overlap: int = 80, model: str = "gpt-4o"):
        self.size = size
        self.overlap = overlap
        self.model = model
    
    def chunk(self, doc: Document) -> list[DocumentChunk]:
        enc = get_encoding(self.model)
        
        # Concatenate all blocks (using embed-friendly text — semantic_content for tables)
        full_text_parts = []
        # Each entry: (char_start, char_end, block_id, page_number)
        block_pos: list[tuple[int, int, str, int | None]] = []
        cursor = 0
        for b in doc.blocks:
            txt = b.get_embed_text()
            if not txt:
                continue
            full_text_parts.append(txt)
            block_pos.append((cursor, cursor + len(txt), b.block_id, b.page_number))
            cursor += len(txt) + 2
            full_text_parts.append("\n\n")
        full_text = "".join(full_text_parts)
        
        if not full_text.strip():
            return []
        
        tokens = enc.encode(full_text)
        chunks: list[DocumentChunk] = []
        step = max(1, self.size - self.overlap)
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.size]
            if not chunk_tokens:
                break
            chunk_text = enc.decode(chunk_tokens)
            
            # Map back to source blocks (best-effort by char range)
            chunk_char_start = len(enc.decode(tokens[:i]))
            chunk_char_end = chunk_char_start + len(chunk_text)
            
            overlapping = [
                (bid, pg) for (s, e, bid, pg) in block_pos
                if s < chunk_char_end and e > chunk_char_start
            ]
            source_block_ids = [bid for bid, _ in overlapping]
            # Pick the first overlapping block's page (chunks usually span 1-2 pages)
            page_number = next((pg for _, pg in overlapping if pg is not None), None)
            
            chunks.append(DocumentChunk(
                document_id=doc.document_id,
                text=chunk_text,
                source_block_ids=source_block_ids,
                page_number=page_number,
                metadata={"chunker": self.name, "size": self.size, "overlap": self.overlap},
            ))
            
            if i + self.size >= len(tokens):
                break
        return chunks
