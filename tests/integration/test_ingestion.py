"""End-to-end integration test for IngestionPipeline.

Uses a synthetic PDF (no API key needed — captioner defaults to NoOpCaptioner
when OPENAI_API_KEY is missing).

Validates:
  1. Pipeline returns a Document with sensible blocks
  2. Cache hit on second call (much faster + same result)
  3. max_pages and page_range parameters work
  4. clear_cache works
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import time
import pytest
from src.pipelines.ingestion import IngestionPipeline
from src.captioners.vlm_captioner import NoOpCaptioner
from src.core.cache import CacheBundle


PDF_PATH = Path(__file__).parent.parent.parent / "data" / "uploads" / "synthetic_earnings.pdf"


@pytest.fixture
def pdf_path():
    if not PDF_PATH.exists():
        from tests.integration.make_test_pdf import make_test_pdf
        make_test_pdf(PDF_PATH)
    return PDF_PATH


@pytest.fixture
def pipeline(tmp_path):
    """Pipeline using a fresh temp cache, NoOp captioner (no API needed)."""
    cache = CacheBundle.from_root(tmp_path)
    return IngestionPipeline(
        captioner=NoOpCaptioner(),
        cache=cache,
    )


def test_basic_ingest(pipeline, pdf_path):
    report = pipeline.ingest(pdf_path)
    assert report.document is not None
    assert len(report.document.blocks) > 0
    assert report.document.n_pages == 2
    assert report.parse_cache_hit is False  # first call


def test_block_diversity(pipeline, pdf_path):
    """Verify parser detects multiple block types in the synthetic PDF."""
    report = pipeline.ingest(pdf_path)
    types = {b.block_type for b in report.document.blocks}
    # Should at least have paragraphs
    assert "paragraph" in types or "h1" in types or "h2" in types


def test_cache_hit_on_second_call(pipeline, pdf_path):
    # First call — populates cache
    r1 = pipeline.ingest(pdf_path)
    t0 = time.perf_counter()
    r2 = pipeline.ingest(pdf_path)
    t1 = time.perf_counter() - t0
    
    assert r1.document.title == r2.document.title
    assert len(r1.document.blocks) == len(r2.document.blocks)
    assert r2.parse_cache_hit is True
    assert t1 < 1.0, f"Cache hit should be fast, was {t1:.2f}s"


def test_max_pages_limit(pipeline, pdf_path):
    report = pipeline.ingest(pdf_path, max_pages=1)
    assert report.document.n_pages == 1
    # All blocks should be from page 1
    for b in report.document.blocks:
        if b.page_number is not None:
            assert b.page_number == 1


def test_page_range(pipeline, pdf_path):
    report = pipeline.ingest(pdf_path, page_range=(2, 2))
    # Should have only page 2 blocks
    pages_seen = {b.page_number for b in report.document.blocks if b.page_number}
    assert pages_seen == {2}


def test_max_pages_creates_separate_cache_entry(pipeline, pdf_path):
    """Different max_pages should NOT collide in cache."""
    r_full = pipeline.ingest(pdf_path)
    r_partial = pipeline.ingest(pdf_path, max_pages=1)
    # Different cache keys → different content
    assert r_full.document.n_pages == 2
    assert r_partial.document.n_pages == 1


def test_clear_cache(pipeline, pdf_path):
    pipeline.ingest(pdf_path)
    assert pipeline.cache.docs.hits + pipeline.cache.docs.misses > 0
    
    cleared = pipeline.clear_cache()
    assert cleared["docs_cleared"] >= 1


def test_report_summary_renders(pipeline, pdf_path):
    report = pipeline.ingest(pdf_path)
    summary = report.summary()
    assert "Document:" in summary
    assert "Cost" in summary
    assert "Blocks" in summary
