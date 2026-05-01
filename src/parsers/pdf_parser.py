"""
Stage 2: Structural Parser.

Takes the loaded page data (text + tables + images) and produces a list of
DocumentBlocks with semantic types (h1/h2/paragraph/table/image) and
heading_path inheritance.

Heading detection heuristics (no LLM call):
  - Lines that match "ALL CAPS" with multiple words → likely h1
  - Lines that are short, end without period, are bold (we proxy via length)
    → candidates for h2
  - Lines like "X | | Q4 2025 ..." (AMD slide header pattern) → page header
  - Lines like "C O R T E X 2 - B U I L D I N G" (Tesla spaced caps) → h1

This is best-effort. Production systems use LayoutLM or LlamaParse for this.
For the lab, the heuristics work well enough on our 3 corpus docs.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Optional

from src.core.interfaces import BaseParser
from src.core.models import Document, DocumentBlock, BoundingBox
from src.core.cache import file_sha256


# Regex patterns for heading detection
_PAT_AMD_PAGE_HEADER = re.compile(r"^\d+\s*\|\s*\|.*\d{4}")  # "9 | | Q4 2025 FINANCIAL RESULTS"
_PAT_SPACED_CAPS = re.compile(r"^[A-Z](\s[A-Z0-9])+(\s*[-–]\s*[A-Z0-9](\s[A-Z0-9])*)*\s*$")  # "C O R T E X 2"
_PAT_ALL_CAPS_TITLE = re.compile(r"^[A-Z][A-Z0-9 \-&,'\u2013/]{4,}$")
_PAT_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")


class PDFStructuralParser(BaseParser):
    """Stage 2: dict from loader → Document with semantic blocks."""
    
    name = "pdf_structural_v1"
    
    def __init__(
        self,
        heading_max_chars: int = 80,
        merge_hyphenated: bool = True,
    ):
        self.heading_max_chars = heading_max_chars
        self.merge_hyphenated = merge_hyphenated
    
    def parse(self, loaded: dict[str, Any], source_path: str | Path) -> Document:
        source_path = Path(source_path)
        meta = loaded.get("metadata", {})
        
        # Note: dict.get returns "" when title field exists but is blank (Tesla PDF case).
        # We need to treat empty string as missing.
        title = (meta.get("title") or "").strip()
        if not title:
            title = source_path.stem.replace("_", " ").title()
        
        doc = Document(
            title=title,
            source_type="pdf",
            source_path=str(source_path),
            source_hash=file_sha256(source_path),
            n_pages=meta.get("n_pages_loaded", 0),
            page_range=meta.get("page_range"),
        )
        
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        
        for page in loaded["pages"]:
            page_num = page["page_num"]
            
            # 1. Process text blocks (bbox-aware) for headings + paragraphs
            for tb in page["text_blocks"]:
                text = self._clean_text(tb["text"])
                if not text:
                    continue
                
                level = self._detect_heading_level(text)
                bbox = BoundingBox(
                    x0=tb["bbox"][0], y0=tb["bbox"][1],
                    x1=tb["bbox"][2], y1=tb["bbox"][3],
                )
                
                if level is not None:
                    # It's a heading. Update the stack.
                    heading_stack = [h for h in heading_stack if h[0] < level]
                    
                    # Block records the path BEFORE pushing this heading
                    doc.blocks.append(DocumentBlock(
                        block_type=f"h{min(level, 4)}",  # type: ignore
                        text=text,
                        page_number=page_num,
                        bbox=bbox,
                        heading_path=[h[1] for h in heading_stack],
                    ))
                    heading_stack.append((level, text))
                else:
                    # Regular paragraph
                    doc.blocks.append(DocumentBlock(
                        block_type="paragraph",
                        text=text,
                        page_number=page_num,
                        bbox=bbox,
                        heading_path=[h[1] for h in heading_stack],
                    ))
            
            # 2. Tables — preserve as markdown for downstream readability
            for tbl in page["tables"]:
                md_table = self._table_rows_to_markdown(tbl["rows"])
                if not md_table.strip():
                    continue
                doc.blocks.append(DocumentBlock(
                    block_type="table",
                    text=md_table,
                    structured_data={
                        "rows": tbl["rows"],
                        "n_rows": tbl["n_rows"],
                        "n_cols": tbl["n_cols"],
                    },
                    page_number=page_num,
                    bbox=BoundingBox(
                        x0=tbl["bbox"][0], y0=tbl["bbox"][1],
                        x1=tbl["bbox"][2], y1=tbl["bbox"][3],
                    ),
                    heading_path=[h[1] for h in heading_stack],
                ))
            
            # 3. Images — block has empty text, real content fills in at captioning stage
            for img in page["images"]:
                doc.blocks.append(DocumentBlock(
                    block_type="image",
                    text="",
                    structured_data={
                        "width": img["width"],
                        "height": img["height"],
                        "image_bytes": img["bytes"],   # used by VLM captioner
                        "image_index": img["image_index"],
                    },
                    page_number=page_num,
                    heading_path=[h[1] for h in heading_stack],
                ))
        
        return doc
    
    # ---- helpers ----
    @staticmethod
    def _reflow_spaced_caps(text: str) -> str:
        """Collapse 'S U M M A R Y  H I G H L I G H T S' → 'SUMMARY HIGHLIGHTS'.
        
        Many PDFs (e.g., Tesla quarterly updates) render headers with a
        per-letter space for visual emphasis. This destroys the actual word
        for both BM25 tokenization AND for downstream LLM judges (Ragas
        context_precision falsely scores these chunks 0).
        
        Algorithm: split on whitespace preserving the run lengths; collapse
        runs of 3+ single-uppercase-char tokens into a single word. Word
        breaks are: any 2+ space gap, any non-single-cap token in the middle.
        Must be called BEFORE whitespace normalization (depends on multi-space).
        """
        # Split into (token, gap_after) pairs by walking; preserve gap lengths.
        # We rebuild the text as we go.
        parts = re.split(r"(\s+)", text)  # alternates: token, ws, token, ws, ...
        out: list[str] = []
        i = 0
        while i < len(parts):
            tok = parts[i]
            # Look ahead: a "spaced-caps run" is consecutive single-A-Z/0-9 tokens
            # separated by EXACTLY one space.
            if re.fullmatch(r"[A-Z0-9]", tok):
                run = [tok]
                j = i + 1
                while j + 1 < len(parts):
                    gap = parts[j]
                    next_tok = parts[j + 1]
                    if gap == " " and re.fullmatch(r"[A-Z0-9]", next_tok):
                        run.append(next_tok)
                        j += 2
                    else:
                        break
                if len(run) >= 3:
                    out.append("".join(run))
                    i = j
                    continue
            out.append(tok)
            i += 1
        return "".join(out)
    
    def _clean_text(self, text: str) -> str:
        text = text.strip()
        if self.merge_hyphenated:
            text = re.sub(r"-\s*\n\s*", "", text)
        # IMPORTANT: reflow spaced-caps BEFORE collapsing whitespace,
        # because Tesla-style "S U M M A R Y  H I G H L I G H T S" uses
        # double-spaces as word boundaries — collapsing first would lose them.
        text = self._reflow_spaced_caps(text)
        text = re.sub(r"\s+", " ", text)
        return text
    
    def _detect_heading_level(self, text: str) -> Optional[int]:
        """Return heading level 1-3, or None if not a heading."""
        if len(text) > self.heading_max_chars:
            return None
        if _PAT_PAGE_NUMBER.match(text):
            return None
        if _PAT_AMD_PAGE_HEADER.match(text):
            return None  # treat AMD page headers as not-headings (skip)
        if _PAT_SPACED_CAPS.match(text):
            return 1   # Tesla-style "C O R T E X 2" → top-level section
        # All caps with ~3+ words → likely section title
        words = text.split()
        if len(words) >= 2 and _PAT_ALL_CAPS_TITLE.match(text):
            return 1
        # Title-cased short heading without trailing period → h2
        if (
            len(words) >= 2 and len(words) <= 8
            and text[0].isupper()
            and not text.endswith(".")
            and sum(1 for w in words if w[0].isupper() if w) >= max(1, len(words) - 1)
        ):
            return 2
        return None
    
    def _table_rows_to_markdown(self, rows: list[list[str]]) -> str:
        if not rows:
            return ""
        cleaned = [[(c or "").strip().replace("\n", " ") for c in row] for row in rows]
        n_cols = max(len(r) for r in cleaned)
        cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]
        
        out = []
        out.append("| " + " | ".join(cleaned[0]) + " |")
        out.append("| " + " | ".join(["---"] * n_cols) + " |")
        for row in cleaned[1:]:
            out.append("| " + " | ".join(row) + " |")
        return "\n".join(out)
