"""
IngestionPipeline — the main orchestrator.

Usage:
    pipeline = IngestionPipeline()
    report = pipeline.ingest("data/uploads/wells_fargo.pdf")
    print(report.summary())
    doc = report.document
    
With cache (default: enabled):
    First call:  loads PDF, runs VLM, caches everything
    Second call: ~10 ms (just unpickles cached Document)

With max_pages / page_range:
    pipeline.ingest("tesla.pdf", max_pages=5)
    pipeline.ingest("tesla.pdf", page_range=(24, 30))   # Financial Statements only
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

from src.core.interfaces import BaseLoader, BaseParser, BaseCaptioner
from src.core.models import Document, IngestionReport
from src.core.cache import CacheBundle, file_sha256
from src.core.config import settings
from src.observability import CostTracker
from src.loaders.pdf_loader import PyMuPDFLoader
from src.parsers.pdf_parser import PDFStructuralParser
from src.captioners.vlm_captioner import GPT4oCaptioner, NoOpCaptioner


class IngestionPipeline:
    """
    Orchestrates Loader -> Parser -> Captioner with a shared cache + cost tracker.
    
    Args:
        loader: BaseLoader instance (default: PyMuPDFLoader)
        parser: BaseParser instance (default: PDFStructuralParser)
        captioner: BaseCaptioner — defaults to GPT4oCaptioner if API key set,
                   else NoOpCaptioner (so the pipeline never crashes for missing key)
        cache: CacheBundle — pass None to disable all caching
        cost_tracker: pass an existing tracker to accumulate across multiple ingests
    """
    
    def __init__(
        self,
        loader: Optional[BaseLoader] = None,
        parser: Optional[BaseParser] = None,
        captioner: Optional[BaseCaptioner] = None,
        cache: Optional[CacheBundle] = None,
        cost_tracker: Optional[CostTracker] = None,
        cache_root: str | Path = "cache",
    ):
        self.loader = loader or PyMuPDFLoader()
        self.parser = parser or PDFStructuralParser()
        self.cost_tracker = cost_tracker or CostTracker()
        
        # Cache is on by default (decision 2A)
        if cache is None:
            cache = CacheBundle.from_root(
                cache_root, enabled=settings.cache_enabled_default
            )
        self.cache = cache
        
        # Default captioner: VLM if key, else no-op (to avoid crashing without keys)
        if captioner is None:
            if settings.has_openai_key:
                captioner = GPT4oCaptioner(
                    cache=self.cache.vlm,
                    cost_tracker=self.cost_tracker,
                )
            else:
                captioner = NoOpCaptioner()
        self.captioner = captioner
    
    def ingest(
        self,
        source: str | Path,
        max_pages: Optional[int] = None,
        page_range: Optional[tuple[int, int]] = None,
        verbose: bool = False,
    ) -> IngestionReport:
        """Run the full pipeline. Returns a report with the document + stats."""
        source = Path(source)
        t0 = time.perf_counter()
        
        # ---- Layer 1: Document cache lookup ----
        captioner_model = getattr(self.captioner, "model", self.captioner.name)
        doc_key = self.cache.docs.make_key(
            source_path=source,
            parser_name=self.parser.name,
            max_pages=max_pages,
            page_range=page_range,
            captioner_model=captioner_model,
        )
        cached_doc = self.cache.docs.get(doc_key)
        
        if cached_doc is not None:
            if verbose:
                print(f" cache HIT for {source.name} -> {doc_key[:12]}...")
            doc = cached_doc
            wall = time.perf_counter() - t0
            return self._build_report(doc, wall, parse_cache_hit=True)
        
        # ---- Stage 1: Load ----
        if verbose:
            print(f"Loading {source.name}...")
        loaded = self.loader.load(source, max_pages=max_pages, page_range=page_range)
        if verbose:
            n_pages = loaded["metadata"].get("n_pages_loaded", 0)
            print(f"   {n_pages} pages loaded")
        
        # ---- Stage 2: Parse ----
        if verbose:
            print(f"Parsing structure...")
        doc = self.parser.parse(loaded, source_path=source)
        n_text = sum(1 for b in doc.blocks if b.block_type in ("paragraph", "h1", "h2", "h3"))
        n_tables = sum(1 for b in doc.blocks if b.block_type == "table")
        n_images = sum(1 for b in doc.blocks if b.block_type in ("image", "chart", "figure"))
        if verbose:
            print(f"   {n_text} text blocks, {n_tables} tables, {n_images} images")
        
        # ---- Stage 3: Caption (eager VLM, decision 2A) ----
        if (n_tables > 0 or n_images > 0):
            if verbose:
                print(f"Captioning {n_tables + n_images} non-text blocks...")
            self.captioner.caption_all(doc)
        
        # ---- Persist to cache ----
        # Strip image bytes before caching (saves disk + avoids serialization issues)
        for b in doc.blocks:
            if b.structured_data and "image_bytes" in b.structured_data:
                # Keep size info, drop bytes
                size_info = {
                    k: v for k, v in b.structured_data.items() if k != "image_bytes"
                }
                b.structured_data = size_info
        
        self.cache.docs.set(doc_key, doc)
        
        wall = time.perf_counter() - t0
        if verbose:
            print(f"Done in {wall:.1f}s. ${self.cost_tracker.total:.4f} this run.")
        return self._build_report(doc, wall, parse_cache_hit=False)
    
    def _build_report(
        self, doc: Document, wall: float, parse_cache_hit: bool
    ) -> IngestionReport:
        n_text = sum(1 for b in doc.blocks if b.block_type in ("paragraph", "h1", "h2", "h3"))
        n_tables = sum(1 for b in doc.blocks if b.block_type == "table")
        n_images = sum(1 for b in doc.blocks if b.block_type in ("image", "chart", "figure"))
        
        return IngestionReport(
            document=doc,
            n_text_blocks=n_text,
            n_table_blocks=n_tables,
            n_image_blocks=n_images,
            n_chunks=0,    # filled in by chunker, not by ingestion
            parse_cache_hit=parse_cache_hit,
            vlm_cache_hits=self.cache.vlm.hits,
            vlm_cache_misses=self.cache.vlm.misses,
            total_cost_usd=self.cost_tracker.total,
            cost_breakdown=dict(self.cost_tracker.by_stage),
            wall_time_seconds=wall,
        )
    
    def clear_cache(self):
        """Wipe everything in the cache. Use sparingly."""
        n_docs = self.cache.docs.clear()
        n_vlm = self.cache.vlm.clear()
        return {"docs_cleared": n_docs, "vlm_cleared": n_vlm}
