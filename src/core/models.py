"""
Domain model — Document -> Page -> Block -> Chunk hierarchy from S4 §3.2.

These are pure data classes. No I/O, no LLM calls, no LangChain dependency.
That means they can be imported anywhere without side effects.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_serializer


BlockType = Literal[
    "title", "h1", "h2", "h3", "h4",
    "paragraph", "list_item",
    "table", "image", "chart", "figure",
    "code", "header", "footer", "caption",
]


class BoundingBox(BaseModel):
    """PDF-style bounding box: top-left + bottom-right in points."""
    x0: float
    y0: float
    x1: float
    y1: float


class DocumentBlock(BaseModel):
    """
    S4 §3.2 Block — smallest semantic unit after structured parsing.
    
    Key fields:
      - block_type: drives downstream (don't split tables, run VLM on images, etc.)
      - heading_path: ["Tesla Q1 2026", "Financial Summary"] for context enrichment
      - semantic_content: LLM-generated NL summary, used for embedding when raw text
        is unhelpful (tables, images, charts)
      - structured_data: raw original form (markdown table, base64 image, etc.)
        kept separately so we can show it back to the user
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    block_id: str = Field(default_factory=lambda: f"blk_{uuid.uuid4().hex[:12]}")
    block_type: BlockType
    text: str = ""
    semantic_content: Optional[str] = None
    structured_data: Optional[dict[str, Any]] = None
    
    page_number: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    heading_path: list[str] = Field(default_factory=list)
    
    # For image/chart blocks — keep raw bytes (or path) for citation/display
    image_path: Optional[str] = None
    
    def get_embed_text(self) -> str:
        """Text used for embedding. Prefer LLM caption over raw."""
        if self.semantic_content:
            return self.semantic_content
        return self.text
    
    @field_serializer("structured_data")
    def _serialize_structured_data(self, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """
        Replace raw bytes (e.g., image_bytes for VLM) with a safe placeholder when serializing.
        At runtime, code reading block.structured_data["image_bytes"] directly still gets the
        actual bytes — only model_dump() / JSON serialization sees the placeholder.
        Without this, LangSmith tracing or any json.dumps() call would crash on UnicodeDecodeError.
        """
        if not value:
            return value
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, bytes):
                out[k] = f"<bytes:{len(v)}>"
            else:
                out[k] = v
        return out
    
    def display_text(self, max_chars: int = 200) -> str:
        """Short human-readable representation for logging."""
        body = self.text or self.semantic_content or "[empty]"
        body = body.replace("\n", " ").strip()
        if len(body) > max_chars:
            body = body[:max_chars] + "..."
        return f"[{self.block_type}] {body}"


class DocumentChunk(BaseModel):
    """
    S4 §3.2 Chunk — what goes into the vector DB.
    
    Key fields:
      - source_block_ids: reverse pointer to blocks -> enables citation
      - parent_chunk_id: for parent-child small-to-big retrieval (§4.5)
      - text: the actual string that gets embedded (post-chunking transformation)
    """
    chunk_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:12]}")
    document_id: str
    text: str
    
    source_block_ids: list[str] = Field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    heading_path: list[str] = Field(default_factory=list)
    page_number: Optional[int] = None
    
    # Free-form metadata for downstream use (chunk strategy name, position, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """
    S4 §3.2 Document — top of the hierarchy.
    
    A Document is the result of running a Loader + Parser + Captioner on a source file.
    It contains the full block list. It does NOT contain chunks — those are derived
    by running a Chunker on this Document.
    """
    document_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    title: str
    source_type: Literal["pdf", "md", "html", "docx", "json"]
    source_path: Optional[str] = None
    source_hash: Optional[str] = None       # SHA256 of source bytes — cache key
    
    uploaded_by: str = "lab_student"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str = "default"
    
    blocks: list[DocumentBlock] = Field(default_factory=list)
    
    # Metadata about the parsing run
    n_pages: int = 0
    page_range: Optional[tuple[int, int]] = None  # (start, end) if partial
    
    @property
    def text(self) -> str:
        """Concatenated text of all blocks, useful for whole-doc operations."""
        return "\n\n".join(b.text for b in self.blocks if b.text)
    
    def blocks_by_type(self, block_type: BlockType) -> list[DocumentBlock]:
        return [b for b in self.blocks if b.block_type == block_type]
    
    def __len__(self) -> int:
        return len(self.blocks)


class IngestionReport(BaseModel):
    """Result of running an IngestionPipeline. Carries cost + cache stats."""
    document: Document
    
    # Counts
    n_text_blocks: int = 0
    n_table_blocks: int = 0
    n_image_blocks: int = 0
    n_chunks: int = 0
    
    # Cache hits/misses per stage
    parse_cache_hit: bool = False
    vlm_cache_hits: int = 0
    vlm_cache_misses: int = 0
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    
    # Cost (USD)
    total_cost_usd: float = 0.0
    cost_breakdown: dict[str, float] = Field(default_factory=dict)
    
    # Timing
    wall_time_seconds: float = 0.0
    
    def summary(self) -> str:
        lines = [
            f"Document: {self.document.title} ({self.document.n_pages} pages)",
            f" Blocks: {self.n_text_blocks} text, {self.n_table_blocks} tables, {self.n_image_blocks} images",
            f" Chunks: {self.n_chunks}",
            f" VLM cache: {self.vlm_cache_hits} hits / {self.vlm_cache_misses} misses",
            f" Embed cache: {self.embedding_cache_hits} hits / {self.embedding_cache_misses} misses",
            f"Cost: ${self.total_cost_usd:.4f}    time {self.wall_time_seconds:.1f}s",
        ]
        return "\n".join(lines)
