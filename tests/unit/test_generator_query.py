"""
Unit tests for RAGGenerator and QueryPipeline.

These tests don't hit the OpenAI API — we substitute a FakeLLM that returns
deterministic outputs. This is the standard pattern for testing LangChain
components offline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Any
from src.core.models import DocumentChunk
from src.core.interfaces import BaseRetriever, BaseGenerator
from src.generators.rag_generator import (
    RAGGenerator, _build_context, _extract_citations, _NO_RESULT_ANSWER,
)
from src.pipelines.query import QueryPipeline, _classify_query


# =============================================================
# Helpers — build chunks + a fake LLM
# =============================================================
def make_chunks(n=3):
    return [
        DocumentChunk(
            chunk_id=f"chk_{i:03d}",
            document_id="doc_test",
            text=f"This is chunk {i} content. Net income was ${5+i}.0 billion.",
            heading_path=["Test Doc", "Section"],
            page_number=i + 1,
        )
        for i in range(n)
    ]


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {}


class FakeLLM:
    """Returns a fixed answer that includes [^1] [^2] citations."""
    def __init__(self, response: str = "Answer with citation [^1] and another [^2]."):
        self.response = response
        self.last_messages = None
    def invoke(self, messages):
        self.last_messages = messages
        return _FakeMessage(self.response)


# =============================================================
# RAGGenerator tests
# =============================================================
def test_build_context_numbers_chunks():
    chunks = make_chunks(3)
    ctx, mapping = _build_context(chunks)
    assert "[Source 1]" in ctx and "[Source 2]" in ctx and "[Source 3]" in ctx
    assert mapping[1] == "chk_000"
    assert mapping[2] == "chk_001"


def test_extract_citations_finds_numbered_refs():
    answer = "The net income grew [^1]. Diluted EPS rose [^3]."
    mapping = {1: "chk_a", 2: "chk_b", 3: "chk_c"}
    cites = _extract_citations(answer, mapping)
    assert cites == ["chk_a", "chk_c"]


def test_extract_citations_dedupes():
    answer = "Claim one [^1]. Claim two also from [^1]. Claim three [^2]."
    mapping = {1: "chk_a", 2: "chk_b"}
    cites = _extract_citations(answer, mapping)
    assert cites == ["chk_a", "chk_b"]


def test_generator_refuses_on_empty():
    gen = RAGGenerator()
    result = gen.generate("anything", [])
    assert result["refused"] is True
    assert result["answer"] == _NO_RESULT_ANSWER
    assert result["citations"] == []


def test_generator_calls_llm_and_parses_citations():
    gen = RAGGenerator()
    gen._llm = FakeLLM("The net income was $5B [^1] and EPS $1.62 [^2].")
    chunks = make_chunks(3)
    result = gen.generate("What was net income?", chunks)
    
    assert result["refused"] is False
    assert "$5B" in result["answer"]
    assert len(result["citations"]) == 2
    assert result["citations"][0] == "chk_000"
    assert result["citations"][1] == "chk_001"


# =============================================================
# QueryPipeline tests
# =============================================================
def test_classify_query_factual():
    assert _classify_query("What is the net income?") == "factual_lookup"
    assert _classify_query("How much revenue did they have?") == "factual_lookup"
    assert _classify_query("How many shares were repurchased?") == "factual_lookup"


def test_classify_query_analytical():
    assert _classify_query("Compare the segments and analyze trends") == "analytical"
    long_q = "What were the main drivers of net interest income changes and how should I think about that?"
    assert _classify_query(long_q) == "analytical"


class FakeRetriever(BaseRetriever):
    name = "fake"
    def __init__(self, chunks_to_return: list[DocumentChunk]):
        self.chunks_to_return = chunks_to_return
        self.last_k = None
    def index(self, chunks):
        pass
    def retrieve(self, query: str, k: int = 5):
        self.last_k = k
        return self.chunks_to_return[:k]


class FakeGenerator(BaseGenerator):
    name = "fake_gen"
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.last_query = None
        self.last_chunks = None
    def generate(self, query, chunks):
        self.last_query = query
        self.last_chunks = chunks
        return self.response


def test_query_pipeline_factual_uses_quick_k():
    chunks = make_chunks(5)
    retriever = FakeRetriever(chunks)
    generator = FakeGenerator({
        "answer": "factual answer", "citations": ["chk_000"], "refused": False,
        "n_sources_used": 1, "n_sources_retrieved": 3,
    })
    pipeline = QueryPipeline(retriever, generator, quick_k=3, deep_k=8)
    
    result = pipeline.query("What is the net income?")
    
    assert retriever.last_k == 3, "factual queries should use quick_k"
    assert "classify→factual_lookup" in result["stages"]
    assert "quick_retrieve→3" in result["stages"]
    assert result["answer"] == "factual answer"


def test_query_pipeline_analytical_uses_deep_k():
    chunks = make_chunks(10)
    retriever = FakeRetriever(chunks)
    generator = FakeGenerator({
        "answer": "analysis", "citations": ["chk_000", "chk_002"], "refused": False,
        "n_sources_used": 2, "n_sources_retrieved": 8,
    })
    pipeline = QueryPipeline(retriever, generator, quick_k=3, deep_k=8)
    
    result = pipeline.query("Compare and contrast the segment trends across both quarters in detail")
    
    assert retriever.last_k == 8, "analytical queries should use deep_k"
    assert "deep_retrieve→8" in result["stages"]


def test_query_pipeline_refuses_on_no_chunks():
    retriever = FakeRetriever([])
    generator = FakeGenerator({"answer": "shouldn't be called", "citations": []})
    pipeline = QueryPipeline(retriever, generator)
    
    result = pipeline.query("What was Apple's revenue?")
    
    assert result["refused"] is True
    assert "refuse" in result["stages"]
    assert generator.last_query is None, "generator shouldn't run when no chunks"


def test_query_pipeline_draws_mermaid():
    pipeline = QueryPipeline(FakeRetriever([]), FakeGenerator({"answer": ""}))
    diagram = pipeline.draw_mermaid()
    assert isinstance(diagram, str)
    assert len(diagram) > 0
