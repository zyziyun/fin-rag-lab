"""Unit tests for chunkers. Uses a hand-built Document, no PDF / API needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import Document, DocumentBlock
from src.chunkers import FixedSizeChunker, RecursiveChunker, ParentChildChunker


def make_doc():
    """Build a synthetic Document for chunker testing."""
    return Document(
        title="Test Doc",
        source_type="md",
        blocks=[
            DocumentBlock(
                block_type="h1", text="Section 1: Introduction",
                heading_path=[],
            ),
            DocumentBlock(
                block_type="paragraph",
                text="This is a long paragraph that should be split into multiple chunks. " * 20,
                heading_path=["Section 1: Introduction"],
            ),
            DocumentBlock(
                block_type="h2", text="Subsection A",
                heading_path=["Section 1: Introduction"],
            ),
            DocumentBlock(
                block_type="paragraph",
                text="Subsection content. " * 10,
                heading_path=["Section 1: Introduction", "Subsection A"],
            ),
            DocumentBlock(
                block_type="table",
                text="| col1 | col2 |\n| --- | --- |\n| a | b |",
                semantic_content="A small table comparing col1 and col2 values.",
                heading_path=["Section 1: Introduction"],
            ),
        ],
    )


# ===== FixedSizeChunker =====
def test_fixed_returns_chunks():
    doc = make_doc()
    chunker = FixedSizeChunker(size=200, overlap=40)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)

def test_fixed_runnable_invoke():
    """BaseChunker.invoke() (Runnable interface) calls chunk()."""
    doc = make_doc()
    chunker = FixedSizeChunker(size=200, overlap=40)
    via_invoke = chunker.invoke(doc)
    via_chunk = chunker.chunk(doc)
    assert len(via_invoke) == len(via_chunk)


# ===== RecursiveChunker =====
def test_recursive_respects_block_boundaries():
    doc = make_doc()
    chunker = RecursiveChunker(chunk_size=200, overlap=40)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    # Each chunk must come from a single source block
    for c in chunks:
        assert len(c.source_block_ids) == 1

def test_recursive_includes_heading_prefix():
    doc = make_doc()
    chunker = RecursiveChunker(chunk_size=200, overlap=40)
    chunks = chunker.chunk(doc)
    # At least some chunks should have the [Section: ...] prefix
    prefixed = [c for c in chunks if c.text.startswith("[Section:")]
    assert len(prefixed) > 0

def test_recursive_skips_heading_blocks():
    """h1/h2/h3 blocks shouldn't produce their own chunks."""
    doc = make_doc()
    chunker = RecursiveChunker(chunk_size=200, overlap=40)
    chunks = chunker.chunk(doc)
    # The h1 "Section 1: Introduction" shouldn't produce a chunk just for itself
    # (it should appear as heading_path on paragraphs)
    for c in chunks:
        # Chunk text should never be exactly the heading text
        assert "Section 1: Introduction" not in c.text or c.text.startswith("[Section:")


# ===== ParentChildChunker =====
def test_parent_child_two_tier():
    doc = make_doc()
    chunker = ParentChildChunker(parent_size=400, child_size=100)
    parents, children = chunker.chunk_with_parents(doc)
    assert len(parents) > 0
    assert len(children) > len(parents), "should have more children than parents"

def test_parent_child_pointers():
    doc = make_doc()
    chunker = ParentChildChunker(parent_size=400, child_size=100)
    parents, children = chunker.chunk_with_parents(doc)
    parent_ids = {p.chunk_id for p in parents}
    for c in children:
        assert c.parent_chunk_id in parent_ids

def test_parent_child_chunk_method_returns_children():
    """The .chunk() method (used by Runnable) returns children, since those go into vector DB."""
    doc = make_doc()
    chunker = ParentChildChunker(parent_size=400, child_size=100)
    only_children = chunker.chunk(doc)
    _, children_via_full = chunker.chunk_with_parents(doc)
    # Same logical content (lengths)
    assert len(only_children) == len(children_via_full)


# ===== Cross-chunker invariants =====
def test_all_chunkers_set_chunker_metadata():
    """Every chunk should record which chunker produced it."""
    doc = make_doc()
    for chunker in [
        FixedSizeChunker(size=200),
        RecursiveChunker(chunk_size=200),
        ParentChildChunker(),
    ]:
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.get("chunker") == chunker.name


def test_all_chunkers_attach_document_id():
    doc = make_doc()
    for chunker in [
        FixedSizeChunker(size=200),
        RecursiveChunker(chunk_size=200),
        ParentChildChunker(),
    ]:
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.document_id == doc.document_id
