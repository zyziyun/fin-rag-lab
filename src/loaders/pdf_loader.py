"""
PyMuPDF-based PDF loader.

This is Stage 1 of ingestion: bytes → raw structured page data. We extract:
  - Plain text per page
  - Layout-aware text blocks (with bounding boxes) for downstream heading detection
  - Embedded images (extracted as PNG bytes)
  - Native PDF tables (PyMuPDF's table finder) — best-effort

NO LLM calls, NO captioning. That's Stage 3.

Why not LangChain's PyMuPDFLoader?
  LangChain's loader returns one Document per page with concatenated text only.
  We need bbox info, image bytes, and table cells to do structural parsing.
  We use raw PyMuPDF and produce our own dict structure.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF

from src.core.interfaces import BaseLoader


class PyMuPDFLoader(BaseLoader):
    """Stage 1: PDF → page-level dict with text/blocks/images/tables."""
    
    name = "pymupdf"
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        min_image_pixels: int = 10000,   # filter tiny icons, page-numbers-as-images
    ):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.min_image_pixels = min_image_pixels
    
    def load(
        self,
        source: str | Path,
        max_pages: Optional[int] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, Any]:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"PDF not found: {source}")
        
        doc = fitz.open(str(source))
        try:
            n_pages_total = doc.page_count
            
            # Resolve which pages to read
            if page_range is not None:
                start, end = page_range
                start = max(0, start - 1)        # caller passes 1-indexed
                end = min(n_pages_total, end)
                page_indices = list(range(start, end))
            else:
                page_indices = list(range(n_pages_total))
            
            if max_pages is not None:
                page_indices = page_indices[:max_pages]
            
            pages = [self._load_page(doc, i) for i in page_indices]
            
            return {
                "pages": pages,
                "metadata": {
                    "n_pages_total": n_pages_total,
                    "n_pages_loaded": len(pages),
                    "page_range": page_range,
                    "title": doc.metadata.get("title", source.stem),
                    "loader": self.name,
                },
            }
        finally:
            doc.close()
    
    def _load_page(self, doc: "fitz.Document", page_idx: int) -> dict[str, Any]:
        page = doc[page_idx]
        
        # Plain text
        text = page.get_text("text")
        
        # Structured blocks with bbox: [(x0, y0, x1, y1, "text", block_no, block_type), ...]
        # block_type: 0 = text, 1 = image
        blocks_raw = page.get_text("blocks")
        text_blocks = []
        for b in blocks_raw:
            if len(b) >= 7 and b[6] == 0:  # text block
                x0, y0, x1, y1, block_text = b[0], b[1], b[2], b[3], b[4]
                text_blocks.append({
                    "bbox": (x0, y0, x1, y1),
                    "text": block_text.strip(),
                })
        
        # Images
        images = []
        if self.extract_images:
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width * pix.height < self.min_image_pixels:
                        pix = None
                        continue
                    if pix.n - pix.alpha >= 4:  # CMYK → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    images.append({
                        "image_index": img_index,
                        "width": pix.width,
                        "height": pix.height,
                        "bytes": img_bytes,
                    })
                    pix = None
                except Exception:
                    # Skip problematic images, don't fail the whole load
                    continue
        
        # Tables (PyMuPDF's built-in finder — works for native PDFs, not scanned)
        tables = []
        if self.extract_tables:
            try:
                table_finder = page.find_tables()
                for tbl in table_finder.tables:
                    rows = tbl.extract()  # list[list[str]]
                    if rows and any(any(cell for cell in row) for row in rows):
                        tables.append({
                            "bbox": tuple(tbl.bbox),
                            "rows": rows,
                            "n_rows": len(rows),
                            "n_cols": len(rows[0]) if rows else 0,
                        })
            except Exception:
                # find_tables can fail on certain PDFs — degrade gracefully
                pass
        
        return {
            "page_num": page_idx + 1,   # 1-indexed for human consumption
            "text": text,
            "text_blocks": text_blocks,
            "images": images,
            "tables": tables,
        }
