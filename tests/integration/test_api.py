"""
Integration tests for the FastAPI server.

Uses fastapi.testclient.TestClient — no real server needed. Patches
the AppState's retrievers + generator with fakes so we don't hit OpenAI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build an isolated app with mock retrievers + generator + cache in tmp_path."""
    monkeypatch.chdir(tmp_path)
    
    # Generate the synthetic PDF inside tmp_path
    from tests.integration.make_test_pdf import make_test_pdf
    pdf_path = tmp_path / "test.pdf"
    make_test_pdf(pdf_path)
    
    from src.api.server import build_app, AppState
    from src.core.cache import CacheBundle
    from src.captioners.vlm_captioner import NoOpCaptioner
    from src.pipelines.ingestion import IngestionPipeline
    from src.observability import CostTracker
    
    app = build_app()
    
    # Replace state's ingestion pipeline with one that uses NoOpCaptioner (no API key needed)
    state_ref = None
    for route in app.routes:
        if hasattr(route, "endpoint"):
            closure = getattr(route.endpoint, "__closure__", None) or ()
            for cell in closure:
                try:
                    val = cell.cell_contents
                    if isinstance(val, AppState):
                        state_ref = val
                        break
                except ValueError:
                    pass
            if state_ref:
                break
    
    assert state_ref is not None, "Could not find AppState in route closures"
    
    state_ref.cost_tracker = CostTracker()
    state_ref.cache = CacheBundle.from_root(tmp_path / "cache", enabled=True)
    state_ref.ingestion = IngestionPipeline(
        captioner=NoOpCaptioner(),
        cache=state_ref.cache,
        cost_tracker=state_ref.cost_tracker,
    )
    
    # Override ensure_retrievers to use fakes
    from tests.unit.test_generator_query import FakeRetriever, FakeGenerator
    from src.pipelines.query import QueryPipeline
    from src.retrievers import HybridRetriever
    
    def fake_ensure():
        if state_ref.vector is None:
            from src.core.models import DocumentChunk
            
            class _MutableFakeVector(FakeRetriever):
                def __init__(self):
                    super().__init__([])
                def index(self, chunks):
                    self.chunks_to_return = list(chunks)
                def search_with_scores(self, query, k=10):
                    return [(c, 1.0) for c in self.chunks_to_return[:k]]
            
            class _MutableFakeBM25(FakeRetriever):
                def __init__(self):
                    super().__init__([])
                def index(self, chunks):
                    self.chunks_to_return = list(chunks)
                def search_with_scores(self, query, k=10):
                    return [(c, 1.0) for c in self.chunks_to_return[:k]]
            
            state_ref.vector = _MutableFakeVector()
            state_ref.bm25 = _MutableFakeBM25()
            state_ref.hybrid = HybridRetriever(state_ref.vector, state_ref.bm25)
            state_ref.generator = FakeGenerator({
                "answer": "Mocked answer about [^1].",
                "citations": [],  # will be filled by chunk lookup
                "refused": False,
                "n_sources_used": 1,
            })
            state_ref.query_pipeline = QueryPipeline(
                state_ref.hybrid, state_ref.generator,
            )
    
    state_ref.ensure_retrievers = fake_ensure
    
    return TestClient(app), pdf_path


def test_health_endpoint(client):
    test_client, _ = client
    r = test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "n_documents_indexed" in body


def test_ingest_then_query(client):
    test_client, pdf_path = client
    
    # Ingest
    r = test_client.post("/ingest", json={"path": str(pdf_path)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_blocks"] > 0
    assert body["n_chunks"] > 0
    
    # Query
    r2 = test_client.post("/query", json={"question": "What was net income?"})
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert "answer" in body2
    assert body2["n_chunks_retrieved"] >= 0


def test_query_without_ingest_fails(client):
    test_client, _ = client
    r = test_client.post("/query", json={"question": "What?"})
    assert r.status_code == 400


def test_ingest_missing_file_404(client):
    test_client, _ = client
    r = test_client.post("/ingest", json={"path": "/nonexistent.pdf"})
    assert r.status_code == 404
