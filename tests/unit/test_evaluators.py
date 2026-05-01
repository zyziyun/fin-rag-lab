"""Tests for evaluators that don't require API calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import Document, DocumentBlock, DocumentChunk
from src.core.interfaces import BaseRetriever
from src.evaluators.coverage import (
    CoverageDiagnostic, _is_data_dense, _numeric_density, compare_strategies, _NUMBER_TOKEN_RE,
)
from src.evaluators.hallucination import (
    HallucinationReport, ClaimVerdict,
)
from src.chunkers import FixedSizeChunker, RecursiveChunker


# ---- Numeric density / data-dense classification
def test_data_dense_true_for_table_rows():
    # Pure table row: every token is a number → density 1.0 → dense
    assert _is_data_dense("164.7 160.7 139.1 135.6 11.0 11.1 24.6")
    # Mostly numbers with a label → density still high
    assert _is_data_dense("Q4 5.4 1.62 11.7% 14.3%")

def test_data_dense_false_for_prose():
    # Realistic financial prose: long sentences with occasional numbers (dates, dollars).
    # Density well below 20% threshold.
    prose = (
        "On January 14, 2026 the company reported strong results across all "
        "operating segments. The CEO remarked that the quarter benefited from "
        "favorable interest rate dynamics and disciplined expense management throughout."
    )
    assert not _is_data_dense(prose)
    # Pure prose, no numbers at all
    assert not _is_data_dense("This is purely qualitative prose with no quantitative data whatsoever.")

def test_numeric_density_values():
    # 4 number tokens out of 4 → 1.0
    assert _numeric_density("5.4 1.62 12% 1,234") == 1.0
    # 0 of 5 tokens → 0
    assert _numeric_density("entirely qualitative narrative descriptive prose") == 0.0
    # 1 of 6 → ~0.166 (above 15% threshold, just barely)
    density = _numeric_density("Revenue grew strongly during the quarter to $5.4B")
    assert 0.10 < density < 0.20


def test_number_regex_picks_up_money_tokens():
    # The regex now matches tokens individually (not findall on full string)
    tokens = "$5.4 $1.62 12% 1,234".split()
    matched = [t for t in tokens if _NUMBER_TOKEN_RE.match(t)]
    assert len(matched) == 4


# ---- CoverageDiagnostic
class _MockRetriever(BaseRetriever):
    name = "mock"
    def __init__(self, chunks):
        self.chunks = chunks
    def index(self, chunks):
        pass
    def retrieve(self, query: str, k: int = 5):
        return self.chunks[:k]


def test_coverage_diagnostic_basic():
    chunks = [
        # Dense: 5 tokens, 4 number-like → density 0.8 → counts as dense
        DocumentChunk(document_id="d", text="Net 5.4 EPS 1.62 11.7%", page_number=1),
        # Sparse: 7 tokens, 0 number-like → density 0.0 → not dense
        DocumentChunk(document_id="d", text="No numbers in this prose at all", page_number=2),
        # Dense: 4 tokens, 2 number-like → 0.5 → dense
        DocumentChunk(document_id="d", text="ROTCE 14.3% ROE 11.7%", page_number=2),
    ]
    diag = CoverageDiagnostic(_MockRetriever(chunks), k=5)
    results = diag.diagnose(["What was net income?"])
    assert len(results) == 1
    r = results[0]
    assert r.n_retrieved == 3
    # 2 of 3 chunks pass the density threshold
    assert abs(r.pct_dense - (2/3)) < 1e-6
    # avg_numeric_density is also reported
    assert 0 < r.avg_numeric_density < 1
    assert r.n_unique_pages == 2


def test_coverage_diagnostic_empty_results():
    diag = CoverageDiagnostic(_MockRetriever([]), k=5)
    results = diag.diagnose(["q"])
    assert results[0].n_retrieved == 0
    assert results[0].pct_dense == 0.0
    assert results[0].avg_numeric_density == 0.0


def test_coverage_to_dataframe():
    chunks = [DocumentChunk(document_id="d", text="1.0 2.0 3.0", page_number=1)]
    diag = CoverageDiagnostic(_MockRetriever(chunks), k=1).diagnose(["q1", "q2"])
    df = CoverageDiagnostic.to_dataframe(diag)
    assert len(df) == 2
    assert set(df.columns) == {
        "query", "n_retrieved", "pct_dense", "avg_density",
        "n_unique_pages", "avg_chars", "top_snippet",
    }


# ---- compare_strategies
def test_compare_strategies_runs():
    """Verify compare_strategies calls each chunker + retriever_factory."""
    doc = Document(title="d", source_type="md", blocks=[
        DocumentBlock(block_type="paragraph", text="Net income was $5.4 billion. " * 30),
        DocumentBlock(block_type="paragraph", text="Revenue grew 12 percent. " * 30),
    ])
    
    # Factory creates a retriever that just returns whatever chunks were indexed
    class _F(BaseRetriever):
        name = "f"
        def __init__(self):
            self._chunks = []
        def index(self, chunks):
            self._chunks = chunks
        def retrieve(self, query, k=5):
            return self._chunks[:k]
    
    chunkers = {
        "fixed": FixedSizeChunker(size=50, overlap=10),
        "recursive": RecursiveChunker(chunk_size=50, overlap=10),
    }
    df = compare_strategies(
        chunkers, doc, queries=["test query"],
        retriever_factory=lambda: _F(), k=3,
    )
    assert "strategy" in df.columns
    assert set(df["strategy"]) == {"fixed", "recursive"}
    # Each strategy should produce a row per query
    assert len(df) == 2  # 2 strategies × 1 query


# ---- HallucinationReport math
def test_hallucination_report_score():
    claims = [
        ClaimVerdict("a", "entailed", "ok"),
        ClaimVerdict("b", "entailed", "ok"),
        ClaimVerdict("c", "refuted", "wrong"),
        ClaimVerdict("d", "unsupported", "?"),
    ]
    report = HallucinationReport(
        n_claims=4, n_entailed=2, n_refuted=1, n_unsupported=1,
        claims=claims,
    )
    assert report.faithfulness_score == 0.5
    d = report.to_dict()
    assert d["faithfulness_score"] == 0.5
    assert d["n_claims"] == 4


def test_hallucination_report_empty():
    report = HallucinationReport(0, 0, 0, 0, [])
    # No claims = perfect by default (nothing to be wrong about)
    assert report.faithfulness_score == 1.0
