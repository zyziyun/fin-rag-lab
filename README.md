# Fin-RAG-Lab — Production-Grade RAG Pipeline

**Target audience**: AI/ML engineering candidates targeting senior IC roles at large US tech and traditional enterprises (e.g., Wells Fargo / Capco AI Engineer — GenAI / Agentic Systems)

**What you'll build**: A complete, class-based, cache-enabled RAG pipeline that ingests real financial documents (Wells Fargo Q4 2025, Tesla Q1 2026, AMD Q4 2025), captions tables and images with VLM, chunks with multiple strategies, retrieves with hybrid (vector + BM25 + RRF), generates with LangGraph orchestration, and evaluates with Ragas — all observable through LangSmith and exposable as a FastAPI service.

**What you'll be able to say in interviews** (after completing): the JD's GenAI / RAG / LLMOps / Ragas / LangSmith / embeddings keywords map 1:1 to specific files in the repo you built.

---

## Architecture overview

```
                    [ User uploads PDF ]
                            ↓
┌─────────────────────────────────────────────────────────┐
│              IngestionPipeline                          │
│                                                         │
│   Loader -> Parser -> Captioner -> cache(layered)         │
│   (PyMuPDF) (struct) (GPT-4o vision)                    │
└─────────────────────────────────────────────────────────┘
                            ↓
                      Document
                   {blocks: [...]}
                            ↓
                      Chunker
                  (Strategy pattern: 3 implementations)
                            ↓
                  Vector + BM25 indexes
                            ↓
                    HybridRetriever (RRF)
                            ↓
              QueryPipeline (LangGraph state machine)
                            ↓
                     Answer + citations
                            ↓
                  Ragas Evaluator -> metrics
                            ↓
                LangSmith Tracing -> dashboard
```

---

## Repo structure

```
rag-lab/
├── LAB_HANDOUT.md              ← this file
├── requirements.txt
├── .env.example
├── scripts/
│   ├── precompute_cache.py     ← run once, generate shareable cache_bundle
│   └── build_notebooks.py      ← rebuilds .ipynb files from src
├── src/
│   ├── core/                   ← domain types, abstract interfaces, cache
│   │   ├── models.py           ← Document, DocumentBlock, DocumentChunk
│   │   ├── interfaces.py       ← BaseLoader/Parser/Captioner/Chunker/Retriever
│   │   ├── cache.py            ← 3-layer content-addressed cache
│   │   └── config.py           ← Settings (env, models, pricing)
│   ├── loaders/                ← Stage 1: PDF -> page dicts
│   ├── parsers/                ← Stage 2: structured Block detection
│   ├── captioners/             ← Stage 3: VLM captioning of tables/images
│   ├── chunkers/               ← Strategy A/B/D class-based
│   ├── retrievers/             ← Vector / BM25 / Hybrid / RRF
│   ├── generators/             ← Citation-aware RAG generator
│   ├── evaluators/             ← Ragas + custom hallucination + coverage diagnostic
│   ├── pipelines/
│   │   ├── ingestion.py        ← IngestionPipeline orchestrator
│   │   └── query.py            ← LangGraph query state machine
│   ├── observability/          ← CostTracker, LangSmith helpers
│   └── api/                    ← FastAPI demo server (POST /ingest, /query)
├── notebooks/
│   ├── 00_quickstart.ipynb     ← 30 lines of LangChain — "feel the chain"
│   ├── 01_parsing.ipynb        ← Real PDF parsing with VLM captioning
│   ├── 02_chunking.ipynb       ← 3 chunking strategies + CoverageDiagnostic tool
│   ├── 03_retrieval.ipynb      ← Vector + BM25 + RRF + parent-child swap
│   ├── 04_generation.ipynb     ← LangGraph state machine + citation tracing
│   ├── 05_evaluation.ipynb     ← Ragas 4 metrics + claim-level hallucination
│   └── 06_observability_fastapi.ipynb  ← LangSmith tracing + FastAPI demo
├── data/
│   ├── uploads/                ← Drop PDFs here
│   └── golden_set/             ← 30-question financial QA set across 5 categories
├── cache/                      ← Auto-managed by IngestionPipeline
│   ├── docs/                   ← Pickled Document objects
│   ├── vlm/                    ← VLM caption cache
│   └── embeddings/             ← Embedding vectors
└── tests/                      ← 57 tests, all passing
    ├── unit/                   ← Fast, no API
    └── integration/            ← End-to-end with synthetic PDF
```

---

## Setup

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-... LANGSMITH_API_KEY=ls__...

# 3. Download the 3 sample PDFs into data/uploads/
#    (URLs are listed in LAB_HANDOUT.md, footnote 1)
#  - data/uploads/wells_fargo.pdf
#  - data/uploads/tesla.pdf
#  - data/uploads/amd.pdf

# 4. (Optional, recommended) Pre-compute the cache once
python scripts/precompute_cache.py \\
    --inputs data/uploads/wells_fargo.pdf \\
             data/uploads/tesla.pdf \\
             data/uploads/amd.pdf

# 5. Open the first notebook
jupyter lab notebooks/00_quickstart.ipynb
```

If you receive a `cache_bundle.zip` from your instructor, just unzip it
into the repo root and the cache will already be warm.

---

## Lab progression

| # | Notebook | Topic | What you'll demo in interviews |
|---|---|---|---|
| 00 | `quickstart` | 30 lines of LangChain LCEL | "I started with vanilla LangChain so I'd understand what production code is *really* abstracting" |
| 01 | `parsing` | Multi-stage ingestion + VLM | "I built a full ingestion pipeline with VLM captioning of tables and images, content-addressed caching, and per-stage cost tracking" |
| 02 | `chunking` | 3 chunking strategies, comparison | "I benchmarked fixed-size vs recursive vs parent-child chunking, improving Ragas faithfulness from 0.7X -> 0.9X" |
| 03 | `retrieval` | Vector + BM25 + RRF hybrid | "Hybrid retrieval with score-agnostic RRF fusion handles both semantic queries and exact-token queries like 'CET1 ratio' well" |
| 04 | `generation` | LangGraph state machine | "Used LangGraph for branching (short query bypass, retrieve-generate, fallback) — production RAG isn't a linear chain" |
| 05 | `evaluation` | Ragas 4-metric + custom hallucination | "I built a 30-question golden set across 4 categories, measured faithfulness/relevancy/precision/recall with Ragas, plus a custom claim-level hallucination detector for compliance" |
| 06 | `observability` | LangSmith + FastAPI | "Instrumented every node with `@traceable`. Exposed pipeline as POST /ingest and POST /query FastAPI endpoints." |

---

## Design choices (what's worth knowing for interviews)

### Why class-based architecture (not just functions)?

Every stage has a `BaseX` abstract class. Concrete implementations inherit + override. **You can swap any stage** (e.g., LlamaParse instead of PyMuPDF, OpenAI embeddings -> Cohere embeddings) **without touching the pipeline**. This is the textbook **Strategy Pattern** and **Dependency Injection** —  named by 80% of senior interview questions about design.

### Why does `BaseChunker` inherit from `Runnable`?

This makes our chunkers **first-class LangChain ecosystem citizens**. You can do `chunker | embedder | indexer`. You can `.batch()`, `.stream()`, `.invoke()`. But the chunker still has its **own** semantic methods (`chunk()`, `chunk_with_parents()`) — the architecture isn't married to LangChain's evolving API.

### Why content-addressed caching?

Cache keys are SHA256 of inputs. **Three guarantees**:
1. **Correctness**: change input -> change key -> no stale data, ever
2. **Sharable**: zip the cache directory, ship to a teammate, they get instant warm-up
3. **Reproducible**: same inputs -> same cache key on every machine, every time

### Why VLM captioning eagerly at ingestion (not lazily at query)?

In production RAG, **read-side latency is the customer-facing metric**. We pay the VLM cost once at ingest (cached forever) so query latency stays p99 < 1s. Lazy VLM at query time would put a 2-3 second LLM call on the hot path.

### Why LangGraph for query pipeline (not just an LCEL chain)?

Production RAG isn't linear:
- Short factual queries -> top-3 chunks suffice (save cost)
- Long analytical queries -> top-8 with parent-child swap (more context)
- Empty retrieval -> **don't call the LLM** (refuse -> save $ + prevent hallucination)

That's a state machine with conditional edges, which is what `QueryPipeline` is. Every node carries `@traceable` so each branch shows up independently in LangSmith.

### Why a custom `HallucinationDetector` on top of Ragas?

Ragas faithfulness gives one score per answer. Compliance-critical domains (finance, legal) need to point at WHICH claim is unsupported. The custom detector decomposes the answer into atomic claims, verifies each one against the joined context with NLI-style prompts, and returns per-claim verdicts with reasoning. That's actionable in a UI: highlight the unsupported sentence in red.

### Why `compare_strategies` instead of hard-coding "use parent-child"?

The right chunking strategy depends on your query distribution. We ship a tool, not a recommendation. `CoverageDiagnostic` quantifies "did retrieval get the data we needed?" in 30 lines of code. Then `compare_strategies(chunkers, doc, queries, retriever_factory)` lets a team make a measurement-driven decision in 5 minutes on a new corpus. **Tools > defaults** for production engineering.

---

## What we observed on the real PDFs (validated)

The instructor pre-ran ingestion on all 3 PDFs. Numbers and observations from that run:

| Document | Pages | Text blocks | Tables | Images | Time | Cost |
|---|---|---|---|---|---|---|
| Wells Fargo Q4 2025 | 12 | 418 | **0** | 1 | 6s | $0.0004 |
| Tesla Q1 2026 | 31 | 530 | 11 | 10 | 95s | $0.0061 |
| AMD Q4 2025 | 34 | 482 | 21 | 61 | 112s | $0.0187 |
| **Total** | 77 | 1,430 | 32 | 72 | ~3.5 min | **$0.025** |

**Two things worth understanding** (both are teaching moments):

1. **Wells Fargo: 0 tables detected.** PyMuPDF's `find_tables()` cannot recover tables that are laid out as positioned text without an underlying table structure — common for press-release-style PDFs. The financial data IS there, but fragmented across 418 text blocks. This is exactly the failure mode `02_chunking.ipynb` opens with: a generic `CoverageDiagnostic` tool reveals that fixed-size chunking cannot retrieve the table content even when asked direct questions. We don't hard-code "WF misses tables" anywhere — the tool measures it.

2. **AMD: 36 cache hits within itself** (out of 104 lookups). AMD's slide deck reuses the same logo / page-header / footer images on many pages. Content-addressed caching deduplicates them automatically — proving the cache key design is correct without needing to write a test for it.

**Cost in the real classroom**: 30 students × full pipeline run × 3 reruns = **$2.25 for an entire cohort**. The "ship a pre-warmed `cache_bundle.zip` to students" pattern is even more valuable than expected — students get a free first-time experience because the instructor pays $0.025 once.

---

## What this lab is NOT

- Not a teach-LangChain-from-scratch course (we use LangChain components but reorganize them around our own abstractions)
- Not exhaustive coverage of every chunking strategy in the literature (we cover 3; semantic chunking and contextual retrieval are discussed conceptually in `02_chunking`)
- Not a deployment course (we show a FastAPI demo in `06`, but real deployment to AWS/Azure/GCP is its own lab)

---

## Footnotes

1. **Source PDFs** (download manually):
   - Wells Fargo Q4 2025: https://www.wellsfargo.com/assets/pdf/about/investor-relations/earnings/fourth-quarter-2025-earnings.pdf
   - Tesla Q1 2026 Update: https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q1-2026-Update.pdf
   - AMD Q4 2025 Earnings Slides: https://d1io3yog0oux5.cloudfront.net/_b0eb9fe85e9ee1621001cc760a9e1d73/amd/db/841/9223/presentation/AMD+Q4'25+Earnings+Slides+FINAL.pdf
