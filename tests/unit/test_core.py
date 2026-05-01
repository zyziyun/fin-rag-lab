"""Unit tests for core domain types and cache. No API calls needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.core.models import Document, DocumentBlock, DocumentChunk, BoundingBox
from src.core.cache import (
    DocumentCache, VLMCache, CacheBundle, file_sha256, _sha256
)
from src.observability import CostTracker


# ===== Models =====
def test_block_get_embed_text_prefers_semantic():
    b = DocumentBlock(block_type="table", text="raw markdown", semantic_content="nice summary")
    assert b.get_embed_text() == "nice summary"
    
def test_block_get_embed_text_falls_back():
    b = DocumentBlock(block_type="paragraph", text="raw text")
    assert b.get_embed_text() == "raw text"

def test_document_text_concat():
    doc = Document(title="d", source_type="md", blocks=[
        DocumentBlock(block_type="paragraph", text="hello"),
        DocumentBlock(block_type="paragraph", text="world"),
    ])
    assert "hello" in doc.text and "world" in doc.text

def test_document_blocks_by_type():
    doc = Document(title="d", source_type="md", blocks=[
        DocumentBlock(block_type="paragraph", text="p"),
        DocumentBlock(block_type="table", text="t"),
        DocumentBlock(block_type="paragraph", text="p2"),
    ])
    assert len(doc.blocks_by_type("paragraph")) == 2
    assert len(doc.blocks_by_type("table")) == 1


# ===== Cache =====
def test_doc_cache_set_get(tmp_path):
    cache = DocumentCache(cache_dir=tmp_path, namespace="docs")
    doc = Document(title="x", source_type="pdf")
    cache.set("key1", doc)
    got = cache.get("key1")
    assert got is not None
    assert got.title == "x"
    assert cache.hits == 1

def test_doc_cache_miss(tmp_path):
    cache = DocumentCache(cache_dir=tmp_path)
    assert cache.get("nonexistent") is None
    assert cache.misses == 1

def test_doc_cache_disabled(tmp_path):
    cache = DocumentCache(cache_dir=tmp_path, enabled=False)
    doc = Document(title="x", source_type="pdf")
    cache.set("k", doc)
    assert cache.get("k") is None  # disabled means writes are no-ops too

def test_vlm_cache_text_key_stable():
    cache = VLMCache(cache_dir="/tmp/test_vlm")
    k1 = cache.make_key_for_text("hello", "gpt-4o-mini")
    k2 = cache.make_key_for_text("hello", "gpt-4o-mini")
    assert k1 == k2
    k3 = cache.make_key_for_text("hello", "gpt-4o")
    assert k3 != k1, "different models should yield different keys"

def test_vlm_cache_image_key_stable():
    cache = VLMCache(cache_dir="/tmp/test_vlm")
    img = b"fake-png-bytes"
    k1 = cache.make_key_for_image(img, "gpt-4o-mini")
    k2 = cache.make_key_for_image(img, "gpt-4o-mini")
    assert k1 == k2

def test_cache_bundle_from_root(tmp_path):
    bundle = CacheBundle.from_root(tmp_path)
    assert bundle.docs is not None
    assert bundle.vlm is not None
    assert (tmp_path / "docs").exists()
    assert (tmp_path / "vlm").exists()

def test_file_sha256_stable(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("hello world")
    h1 = file_sha256(p)
    h2 = file_sha256(p)
    assert h1 == h2 and len(h1) == 64

def test_doc_cache_clear(tmp_path):
    cache = DocumentCache(cache_dir=tmp_path)
    cache.set("a", Document(title="A", source_type="pdf"))
    cache.set("b", Document(title="B", source_type="pdf"))
    n = cache.clear()
    assert n == 2
    assert cache.get("a") is None


# ===== Cost Tracker =====
def test_cost_tracker_records_llm():
    ct = CostTracker()
    cost = ct.record_llm("test_stage", "gpt-4o-mini", input_tokens=1000, output_tokens=500)
    # gpt-4o-mini: $0.000150/1k input + $0.000600/1k output
    expected = 1000/1000 * 0.000150 + 500/1000 * 0.000600
    assert abs(cost - expected) < 1e-9
    assert ct.total > 0
    assert ct.n_calls["test_stage"] == 1

def test_cost_tracker_handles_unknown_model():
    ct = CostTracker()
    cost = ct.record_llm("s", "unknown-model", 1000, 500)
    assert cost == 0.0
    assert ct.n_calls["s"] == 1  # call still counted

def test_cost_tracker_vlm_image():
    ct = CostTracker()
    ct.record_vlm_image()
    assert ct.total > 0
    assert ct.n_calls["vlm_caption"] == 1
