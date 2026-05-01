"""Builds the lab notebooks. Idempotent - re-run anytime to refresh."""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def build(cells, path: Path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        nbf.write(nb, f)
    rel = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
    print(f"  built {rel}")


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


# =============================================================
# 00_quickstart.ipynb - pure LangChain, "feel the chain"
# =============================================================
def build_00_quickstart():
    cells = [
        md("""# 00 - Quickstart: Build a RAG in 30 lines

**Goal**: Feel the "compositional" nature of LangChain - turning a PDF into a
question-answering system takes just 5 components chained with `|`.

**Document**: `data/uploads/wells_fargo.pdf` (Wells Fargo Q4 2025 earnings release)

**Estimated time**: 10 minutes + a few API calls (~$0.01)

---

> Before you start: download the Wells Fargo Q4 2025 earnings PDF to
> `data/uploads/wells_fargo.pdf`:
>
> https://www.wellsfargo.com/assets/pdf/about/investor-relations/earnings/fourth-quarter-2025-earnings.pdf
"""),
        md("""## Step 1: Five imports is all you need

A complete RAG needs: **Loader + Splitter + Embeddings + VectorStore + LLM**.
LangChain provides production-ready implementations for each.
"""),
        code("""from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load OPENAI_API_KEY from .env
from dotenv import load_dotenv
load_dotenv()

print("All imports loaded.")"""),
        md("""## Step 2: Load - turn the PDF into LangChain Documents"""),
        code("""docs = PyMuPDFLoader("../data/uploads/wells_fargo.pdf").load()
print(f"Loaded {len(docs)} pages")
print(f"\\nFirst 200 characters of page 1:")
print(docs[0].page_content[:200])"""),
        md("""## Step 3: Split - cut into retrievable chunks"""),
        code("""splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")
print(f"\\nFirst chunk:\\n{chunks[0].page_content[:200]}")"""),
        md("""## Step 4: Index - load into Chroma with OpenAI embeddings"""),
        code("""embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Test
results = retriever.invoke("What was Wells Fargo's net income?")
print(f"Retrieved {len(results)} relevant chunks")
print(f"\\nTop chunk:\\n{results[0].page_content[:300]}")"""),
        md("""## Step 5: Wire everything into a chain - the "feel" moment

The `|` operator is the heart of LangChain Expression Language (LCEL).
It lets you connect components like Lego bricks:

- `retriever` turns a query into a list of Documents
- `prompt` renders (context, question) into a prompt
- `llm` turns the prompt into a response
- `StrOutputParser` turns the response into a string

Reading the chain almost reads like English.
"""),
        code("""prompt = ChatPromptTemplate.from_template(\"\"\"You are a financial analyst assistant.
Answer the question based ONLY on this context. If the answer isn't there, say so.

Context:
{context}

Question: {question}

Answer:\"\"\")

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# This is the chain. It reads almost like English.
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Chain built. Run a few queries:")"""),
        md("""## Step 6: Ask questions - watch it work"""),
        code("""questions = [
    "What was Wells Fargo's Q4 2025 net income?",
    "How did the Consumer Banking segment perform?",
    "What is the CET1 ratio?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {chain.invoke(q)}\\n")
    print("-" * 60)"""),
        md("""## But pause - ask yourself these questions

Those 30 lines look perfect. You can ask any question, get a reasonable answer.
**But if you were the Capco interviewer, here's what they'd ask:**

| Question | Can you answer? |
|---|---|
| Is `chunk_size=500` optimal? How do you know? | ? |
| What happened to the 8 financial tables in the PDF? (Hint: mostly thrown away) | ? |
| When the system answers wrong, how do you debug? | ? |
| Are the answers really from the PDF, or is the LLM making things up? How do you test? | ? |
| What happens to performance when you go from 1 doc to 100? | ? |
| Once it's deployed, how do you monitor latency, cost, quality? | ? |

**These are exactly the questions the next 6 notebooks answer.**

| Notebook | Topic | Question it answers |
|---|---|---|
| 01_parsing | Real PDF parsing (tables + images + VLM caption) | "What happened to the tables?" |
| 02_chunking | Three chunking strategies compared | "Is chunk_size=500 optimal?" |
| 03_retrieval | Vector + BM25 + RRF hybrid | "Why is hybrid retrieval better?" |
| 04_generation | LangGraph orchestration + citations | "How do I debug?" |
| 05_evaluation | Ragas + 30-question golden set | "Are the answers true? How do I test?" |
| 06_observability | LangSmith + FastAPI deployment | "How do I monitor in production?" |

Ready? Open `01_parsing.ipynb`.
"""),
    ]
    build(cells, NOTEBOOKS_DIR / "00_quickstart.ipynb")


# =============================================================
# 01_parsing.ipynb - using the new architecture
# =============================================================
def build_01_parsing():
    cells = [
        md("""# 01 - Parsing: How PDFs really get processed

**Aligns with**: S4 Sec. 3.2 and 3.3 | **Estimated time**: 30 minutes | **Estimated cost**: ~$0.10

## What 00_quickstart left unanswered

In `00_quickstart` we used `PyMuPDFLoader` to load the Wells Fargo PDF in one
line. But **do the chunks you retrieve actually contain complete information?**

Let's run an experiment to see what the quickstart RAG **cannot** answer.
"""),
        md("""## Step 1: See what the quickstart missed

Ask a question whose answer **only exists in a table**: what is ROTCE?
"""),
        code("""# Path setup: notebook is in notebooks/, src/ lives at ../src/
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()"""),
        code("""# Reproduce the quickstart RAG (default PyMuPDFLoader behavior)
from langchain_community.document_loaders import PyMuPDFLoader

raw_docs = PyMuPDFLoader(str(ROOT / "data/uploads/wells_fargo.pdf")).load()

# What does PyMuPDFLoader give us for page 1?
print("=" * 60)
print("PyMuPDFLoader page 1 output (first 1500 chars)")
print("=" * 60)
print(raw_docs[0].page_content[:1500])"""),
        md("""Notice anything? **The tables are squashed into a mess.** The Selected Income
Statement Data table - every number, every header - is jammed together with
surrounding text.

Worse: **images are gone entirely**. For something like a Tesla deck with factory
photos, PyMuPDFLoader treats those pages as if they only contain a few caption lines.

**This is why we need to rewrite the parsing layer.**
"""),
        md("""## Step 2: Our industrial-grade ingestion pipeline

Introduce `IngestionPipeline` - it does three things LangChain's defaults skip:

1. **Structural parsing**: detects heading / paragraph / table / image and
   **preserves hierarchy**
2. **VLM caption**: tables and images get a natural-language summary from a
   vision-capable model, used for embedding
3. **Three-layer cache**: document / VLM / embedding cache, all content-addressed,
   so **a student rerunning the lab pays $0**

The architecture:

```
[ PDF bytes ]
      |
   Loader (PyMuPDF)         <- Stage 1: extract raw text + tables + images
      |
   Parser (Structural)      <- Stage 2: detect block types, build heading_path
      |
   Captioner (VLM)          <- Stage 3: tables/images -> semantic_content
      |
   Document {blocks: [...]}
```

Each stage is an **implementation of an abstract base class**
(`BaseLoader` / `BaseParser` / `BaseCaptioner`), independently swappable and
mockable in tests.
"""),
        code("""from src.pipelines.ingestion import IngestionPipeline

# Default config: PyMuPDFLoader + PDFStructuralParser + GPT5MiniCaptioner + 3-layer cache (on)
pipeline = IngestionPipeline()

# One-line ingestion
report = pipeline.ingest(
    str(ROOT / "data/uploads/wells_fargo.pdf"),
    verbose=True,
)
print()
print(report.summary())"""),
        md("""**Key observations**:

- `n_text_blocks` is much larger than `len(raw_docs)` - we split each page into
  multiple semantic blocks
- `n_table_blocks` > 0 means **tables were detected** as standalone blocks
  (note: WF press releases use positioned-text tables; PyMuPDF's table detector
  may report 0 here, which is itself a useful signal we'll work with in 02)
- Run this cell a second time: `parse_cache_hit=True`, 0 seconds, $0 spent
"""),
        md("""## Step 3: Inspect the Document structure"""),
        code("""doc = report.document
print(f"Title: {doc.title}")
print(f"  {doc.n_pages} pages, {len(doc.blocks)} total blocks")
print(f"  source_hash: {doc.source_hash[:16]}...")
print()

# Counts by block type
from collections import Counter
type_counts = Counter(b.block_type for b in doc.blocks)
for t, n in type_counts.most_common():
    print(f"  {t:12s}: {n}")"""),
        code("""# Look at the first 5 blocks
print("First 5 blocks:")
for b in doc.blocks[:5]:
    print(f"  [{b.block_type}] page {b.page_number}, heading_path={b.heading_path}")
    print(f"      {b.display_text(max_chars=120)}")"""),
        md("""## Step 4: What VLM captioning does - find a table block

Every `table` block has a `semantic_content` field, which is a natural-language
summary generated by a vision-capable LLM. Example:
"""),
        code("""# Find the first table block
table_blocks = doc.blocks_by_type("table")
print(f"Found {len(table_blocks)} table blocks\\n")

if table_blocks:
    tb = table_blocks[0]
    print(f"Heading path: {' > '.join(tb.heading_path) or '(root)'}")
    print(f"Page: {tb.page_number}")
    print()
    print("=" * 60)
    print("Raw table (markdown format, stored in .text)")
    print("=" * 60)
    print(tb.text[:600])
    print()
    print("=" * 60)
    print("VLM-generated semantic_content (used for embedding)")
    print("=" * 60)
    print(tb.semantic_content if tb.semantic_content else "[no caption - probably no API key]")"""),
        md("""**Why does this matter?**

Embedding a blob of `| col | col | col |` characters directly produces a
near-meaningless vector that won't match queries.

Embedding the NL summary "*Wells Fargo Q4 2025 Selected Income Statement Data
showing revenue, net income, EPS comparing Q4 2024 vs Q4 2025...*" makes vector
search instantly able to match queries like "what was Wells Fargo's revenue
change Q4".

**This is the caption-then-embed pattern from S4 Sec. 3.3.**
"""),
        md("""## Step 5: Cache in action - second run is $0

Test the cache:
"""),
        code("""import time

# Second run (already ingested above, should be a cache hit)
t0 = time.perf_counter()
report1 = pipeline.ingest(str(ROOT / "data/uploads/wells_fargo.pdf"))
t1 = time.perf_counter() - t0
print(f"Second run on same file:")
print(f"  Wall time: {t1*1000:.1f} ms")
print(f"  Cost: ${report1.total_cost_usd:.4f}")
print(f"  Cache hit: {report1.parse_cache_hit}")"""),
        md("""## Step 6: page_range parameter - process specific pages only

Financial Statements typically appear in the last few pages of an earnings PDF.
If you only care about the numbers, you can ingest just those pages, saving
VLM cost and time.
"""),
        code("""# Process only the first 3 pages
report_partial = pipeline.ingest(
    str(ROOT / "data/uploads/wells_fargo.pdf"),
    max_pages=3,
)
print(f"First 3 pages: {report_partial.document.n_pages} pages, {len(report_partial.document.blocks)} blocks")

# Note: cache key includes max_pages, so this won't collide with the full-doc cache
report_full = pipeline.ingest(str(ROOT / "data/uploads/wells_fargo.pdf"))
print(f"Full file: {report_full.document.n_pages} pages, {len(report_full.document.blocks)} blocks")"""),
        md("""## Step 7: Drop in your own PDF

The whole pipeline is designed for **"any PDF in, structured Document out"**.
Switching documents is just a path change:

```python
# For example, Tesla Q1 2026
report = pipeline.ingest("data/uploads/tesla.pdf")

# Or a Microsoft 10-K for an interview demo
report = pipeline.ingest("/path/to/msft_10k.pdf")
```

**This is what production ingestion looks like** - not a hard-coded file but a
service that ingests arbitrary PDFs. In `06_observability` we wrap this as a
FastAPI endpoint and turn it into a real service.
"""),
        md("""## Exercise 1.A

Ingest the Wells Fargo PDF with **different max_pages** values:

1. `max_pages=2` (first 2 pages only)
2. `max_pages=5` (first 5 pages)
3. Unbounded (all ~10 pages)

Verify:
- The 3 cache keys are **independent** (no cross-pollution)
- After running unbounded once, running with `max_pages=2` is **still a fresh run**
  (different cache key)
- After `clear_cache()`, all 3 versions need to re-run
"""),
        code("""# TODO: your exercise code
# pipeline.clear_cache()  # clear
# report_a = pipeline.ingest(..., max_pages=2)
# report_b = pipeline.ingest(..., max_pages=5)
# report_c = pipeline.ingest(...)
# Then run the same three again and check parse_cache_hit is True
"""),
        md("""## Recap of the core concepts

| Concept | Where it lives |
|---|---|
| **S4 Sec. 3.2 Document -> Block hierarchy** | `Document.blocks: list[DocumentBlock]`, each Block has `block_type`, `heading_path`, `bbox` |
| **S4 Sec. 3.3 Caption-then-embed** | `Captioner` adds `semantic_content` to table/image blocks; `block.get_embed_text()` prefers caption |
| **Strategy pattern** | `BaseLoader` / `BaseParser` / `BaseCaptioner` abstract classes with multiple implementations |
| **Content-addressed cache** | Cache key = file SHA256 + parser config + captioner model. Change any input -> invalidate |
| **Cost transparency** | `report.total_cost_usd` and `report.cost_breakdown` make every dollar visible |

Next, `02_chunking.ipynb` answers the other quickstart question:
**Is chunk_size=500 optimal? How do you know?**
"""),
    ]
    build(cells, NOTEBOOKS_DIR / "01_parsing.ipynb")


# =============================================================
# 02_chunking.ipynb - uses CoverageDiagnostic for the WF reveal
# =============================================================
def build_02_chunking():
    cells = [
        md("""# 02 - Chunking: Three strategies + a general retrieval coverage tool

**Aligns with**: S4 Sec. 4 | **Estimated time**: 45 minutes | **Estimated cost**: ~$0.05

## The question 00_quickstart left unanswered: is chunk_size=500 optimal? How do you know?

Two things in this notebook:

1. **Build a general tool**, `CoverageDiagnostic` - quantify "did your retrieval
   actually pull the relevant content?" for any corpus, any chunker
2. **Use the tool** to compare 3 chunking strategies on WF / Tesla / AMD

> **Design choice**: we did NOT hard-code the fact "WF lost 8 tables" as a demo.
> We wrote a diagnostic function that runs on **any corpus**, then **used WF as
> the example**. So when a student switches documents (e.g., a Microsoft 10-K
> for an interview demo), the whole notebook still applies.
"""),
        code("""import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()

from src.pipelines.ingestion import IngestionPipeline
from src.chunkers import FixedSizeChunker, RecursiveChunker, ParentChildChunker
from src.retrievers import VectorRetriever
from src.evaluators.coverage import CoverageDiagnostic, compare_strategies
import pandas as pd"""),
        md("## Step 1: Ingest Wells Fargo (cache hit, should return instantly)"),
        code("""pipeline = IngestionPipeline()
report = pipeline.ingest(str(ROOT / "data/uploads/wells_fargo.pdf"))
doc = report.document
print(f"Title: {doc.title}")
print(f"  {len(doc.blocks)} blocks total")
print(f"  {report.n_table_blocks} table blocks (note: PyMuPDF may report 0 because WF tables are positioned text, not native table structures)")
print(f"  parse_cache_hit = {report.parse_cache_hit}")"""),
        md("## Step 2: Three chunkers, all class-based"),
        code("""chunkers = {
    "fixed_size_500":  FixedSizeChunker(size=500, overlap=80),
    "recursive_400":   RecursiveChunker(chunk_size=400, overlap=60),
    "parent_child":    ParentChildChunker(parent_size=800, child_size=150),
}

for name, ch in chunkers.items():
    chunks = ch.chunk(doc)
    avg = sum(len(c.text) for c in chunks)//max(1,len(chunks))
    print(f"  {name:18s}: {len(chunks):4d} chunks (avg {avg} chars)")"""),
        md("""## Step 3: CoverageDiagnostic - what fixed_size actually does on WF

`embeddings_cache_dir` makes the second run free:
"""),
        code("""test_queries = [
    "What was Wells Fargo's Q4 2025 net income?",
    "What was the diluted EPS in Q4 2025?",
    "What was the CET1 ratio?",
    "What was the average loan balance?",
    "How did Consumer Banking perform?",
    "What were the main drivers of net interest income?",
    "What capital return actions were taken?",
    "How did the segment compositions change?",
]

chunks_fixed = chunkers["fixed_size_500"].chunk(doc)
retriever = VectorRetriever(
    persist_dir=str(ROOT / "tmp_chroma_fixed"),
    collection="lab_02_fixed",
    embeddings_cache_dir=ROOT / "cache/embeddings",
)
retriever.reset()
retriever.index(chunks_fixed)

diag = CoverageDiagnostic(retriever, k=5).diagnose(test_queries)
df = CoverageDiagnostic.to_dataframe(diag)
df"""),
        md("""**Two columns to focus on**:

- `pct_dense`: of the 5 retrieved chunks, **how many are "data-dense"** (>=20% of
  tokens are numeric - typical of table fragments)
- `avg_density`: average **numeric density** across retrieved chunks (0 = pure
  prose, 1 = all numbers)

**How to read this**: for a fact-style query like "Q4 2025 net income?", you want
**high `pct_dense` (>0.4) and high `avg_density` (>0.15)** - that means the
retriever found the table region. If both metrics are low (<=0.1), retrieval
returned a pile of prose paragraphs and **the chunker shredded the table**.

> **Why not just check "does it contain a number"?** Because every paragraph in
> a financial doc has dates and dollar figures - "contains a number" is almost
> always True. **Density** is the actual signal for whether the chunker preserved
> the table structure. The metric evolution itself is a real engineering lesson
> (start simple, observe saturation, refine).

On WF, fixed_size typically has notably lower `avg_density` than recursive /
parent_child on numeric queries - the positioned-text tables get shredded
across multiple chunks, losing their density.
"""),
        md("## Step 4: `compare_strategies` - direct horizontal comparison of all 3 chunkers"),
        code("""def make_retriever():
    \"\"\"Factory: returns a fresh VectorRetriever with embedding cache.\"\"\"
    import uuid
    return VectorRetriever(
        persist_dir=str(ROOT / f"tmp_chroma_compare_{uuid.uuid4().hex[:6]}"),
        collection=f"lab_02_compare_{uuid.uuid4().hex[:6]}",
        embeddings_cache_dir=ROOT / "cache/embeddings",
    )

result_df = compare_strategies(
    chunkers=chunkers,
    doc=doc,
    queries=test_queries,
    retriever_factory=make_retriever,
    k=5,
)

pivot = result_df.pivot_table(
    values="avg_density", index="query", columns="strategy", aggfunc="first",
).round(2)
pivot"""),
        md("## Step 5: Same tool applied to Tesla / AMD"),
        code("""tesla_doc = pipeline.ingest(str(ROOT / "data/uploads/tesla.pdf")).document

tesla_queries = [
    "What was Tesla's Q1 2026 vehicle production?",
    "What was the energy storage deployment?",
    "How did automotive margins evolve?",
    "What is the AI/Cortex initiative?",
]

tesla_compare = compare_strategies(
    chunkers=chunkers, doc=tesla_doc, queries=tesla_queries,
    retriever_factory=make_retriever, k=5,
)
tesla_compare.pivot_table(
    values="avg_density", index="query", columns="strategy", aggfunc="first"
).round(2)"""),
        md("""## Step 6: Decision framework

**There is no "best chunking strategy" - only "the best strategy for your query
distribution":**

| Query type | Recommended | Why |
|---|---|---|
| Short numeric lookup | parent_child | Small child precisely hits the block with the number |
| Long narrative | recursive | Preserves heading + section boundaries |
| Generic baseline | recursive | Simple and reliable |
| **Avoid** | fixed_size | Cuts across blocks, drops context |

**Interview talking point**:

> "I didn't hard-code a chunking strategy. I built `compare_strategies` plus the
> `CoverageDiagnostic` quantification function so the team can decide which
> chunker fits a new corpus in 5 minutes. This is measurement-driven engineering."

## Exercise 2.A

Add the AMD PDF (image-heavy) to `compare_strategies`. Predict and verify:
- AMD's PowerPoint style means repeated images, so embedding cache hit rate is high
- AMD data is scattered through charts, so fixed_size does worse on numeric queries
"""),
        code("""# TODO: amd_doc = pipeline.ingest(...)
# TODO: amd_compare = compare_strategies(chunkers, amd_doc, [...], make_retriever)
"""),
    ]
    build(cells, NOTEBOOKS_DIR / "02_chunking.ipynb")


# =============================================================
# 03_retrieval.ipynb - vector vs BM25 vs hybrid
# =============================================================
def build_03_retrieval():
    cells = [
        md("""# 03 - Retrieval: Vector + BM25 + Hybrid (RRF)

**Aligns with**: S4 Sec. 4.4 | **Estimated time**: 30 minutes | **Estimated cost**: ~$0.02

## The question 02 left

In `02_chunking` we used vector retrieval throughout. But vector retrieval has
known weaknesses on proper nouns / acronyms / exact strings:

| Query | Vector performance | Why |
|---|---|---|
| "What was net income?" | Good | Natural-language match |
| "What was the **CET1** ratio?" | Mediocre | Acronym embedding is imprecise |
| "`Net Charge-offs`?" | Poor | Needs exact-string match |

**Solution: Hybrid retrieval** - vector + BM25 each retrieve independently,
then **Reciprocal Rank Fusion (RRF)** merges the results.

## What this notebook answers (with real data)

1. When does cross-document retrieval **leak across docs**? How do you fix it?
2. Vector vs BM25 vs Hybrid - which wins on which query types?
3. What is RRF actually doing internally?

> **Important**: This notebook does NOT pre-write the conclusion before showing
> you data. We made that mistake in 02 (pre-baked "hybrid wins"). This time the
> data speaks for itself.
"""),
        code("""import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()

from src.pipelines.ingestion import IngestionPipeline
from src.chunkers import RecursiveChunker
from src.retrievers import VectorRetriever, BM25Retriever, HybridRetriever
import pandas as pd"""),
        md("## Step 1: Ingest 3 docs and chunk (note: `document_id` is auto-stored in metadata)"),
        code("""pipeline = IngestionPipeline()
docs = []
for name in ["wells_fargo", "tesla", "amd"]:
    report = pipeline.ingest(str(ROOT / f"data/uploads/{name}.pdf"))
    docs.append(report.document)
    print(f"  {report.document.title[:50]:50s} doc_id={report.document.document_id[:16]}")

# Track each doc's id - we'll need it for filtering later
DOC_IDS = {"wf": docs[0].document_id, "tesla": docs[1].document_id, "amd": docs[2].document_id}

chunker = RecursiveChunker(chunk_size=400, overlap=60)
all_chunks = []
for doc in docs:
    all_chunks.extend(chunker.chunk(doc))
print(f"\\nTotal chunks across 3 docs: {len(all_chunks)}")"""),
        md("## Step 2: Index into 3 retrievers (one shared collection, 3 docs mixed)"),
        code("""vec = VectorRetriever(
    persist_dir=str(ROOT / "tmp_chroma_03"),
    collection="lab_03",
    embeddings_cache_dir=ROOT / "cache/embeddings",
)
vec.reset(); vec.index(all_chunks)
print("  Vector indexed")

bm = BM25Retriever()
bm.index(all_chunks)
print("  BM25 indexed")

hyb = HybridRetriever(vector=vec, bm25=bm)
print("  Hybrid composed")"""),
        md("""## Step 3: Cross-document leakage - real bug demonstration

We placed all 3 docs' chunks into one collection. When asking a WF question, the
retriever may pull Tesla content - because BM25's keyword matching has no notion
of "which document".

The next cell labels each retrieved top-3 chunk with its source doc. **Watch
the `from_doc` column: ideal is `wf`, leakage shows up as `tesla` or `amd`.**
"""),
        code("""def lookup_doc(chunk):
    \"\"\"Reverse-lookup: which doc did this chunk come from?\"\"\"
    for k, v in DOC_IDS.items():
        if chunk.document_id == v:
            return k
    return "?"

q = "What were the main drivers of NII change?"   # WF-specific question
print(f"Query: {q!r}\\n  (Wells Fargo's metric - Net Interest Income)\\n")

rows = []
for r_name, retriever in [("vector", vec), ("bm25", bm), ("hybrid", hyb)]:
    chunks = retriever.retrieve(q, k=3)
    for rank, c in enumerate(chunks, 1):
        rows.append({
            "retriever": r_name, "rank": rank,
            "from_doc": lookup_doc(c),
            "page": c.page_number,
            "snippet": c.text[:70].replace("\\n", " ") + "...",
        })
pd.DataFrame(rows)"""),
        md("""**How to read this table**:

- All `from_doc` values are `wf` -> retriever didn't leak
- Any `tesla` or `amd` rows -> **bug**: the retriever pulled in another doc's chunks

This isn't the retriever's "fault". We never told it **which document to search
within**. Production RAG must filter by `document_id` / `tenant_id`, otherwise
one customer's private documents show up in another customer's results
(a compliance incident).
"""),
        md("""## Step 4: The fix - Chroma's `where` filter (production-grade)

ChromaDB already stores `document_id` as metadata at index time
(see `src/retrievers/vector.py` lines 61-67). At query time, pass
`filter={'document_id': wf_id}` for server-side scoping.
"""),
        code("""# Use Chroma store's lower-level API directly. In production, wrap this in
# VectorRetriever.retrieve(filter=...).
wf_filter = {"document_id": DOC_IDS["wf"]}

q = "What were the main drivers of NII change?"
results_filtered = vec.store.similarity_search_with_score(q, k=3, filter=wf_filter)

print(f"Query: {q!r}\\n  (with WF filter)\\n")
for rank, (lc_doc, score) in enumerate(results_filtered, 1):
    src_id = lc_doc.metadata.get("document_id", "")
    src = next((k for k, v in DOC_IDS.items() if v == src_id), "?")
    print(f"  {rank}. from_doc={src}  page={lc_doc.metadata.get('page_number')}  score={score:.3f}")
    print(f"     {lc_doc.page_content[:80].strip()}...")"""),
        md("""**Core takeaways**:
- Cross-document leakage is a real bug - **default behavior is unsafe**
- Chroma's `where` filter is server-side (efficient); BM25 (in-memory) needs
  client-side post-filter (less efficient but simple)
- Production setups often go further: **one collection per tenant or per doc**
  for physical isolation

For the rest of the comparisons we'll use WF-scoped chunks (fair comparison +
avoids the leakage problem).
"""),
        code("""# Rebuild retrievers indexed only on WF chunks (so the benchmark is apples-to-apples)
wf_chunks = [c for c in all_chunks if c.document_id == DOC_IDS["wf"]]
print(f"WF-only chunks: {len(wf_chunks)}")

vec.reset(); vec.index(wf_chunks)
bm = BM25Retriever(); bm.index(wf_chunks)
hyb = HybridRetriever(vector=vec, bm25=bm)
print("  All 3 retrievers re-indexed on WF only")"""),
        md("""## Step 5: Three retrievers head-to-head - look at top-1

We're not using an aggregate metric (we learned in 02 that single metrics
mislead). Instead, **inspect the top-1 chunk each retriever returns** and judge
relevance directly.

For each query we add a `keyword_hit` column: does the top-1 chunk contain the
query's keywords (case-insensitive)? It's a **cheap honest signal** - imperfect,
but less misleading than aggregates.
"""),
        code("""bench_queries = [
    # (query, query_type, key_keywords_to_check)
    ("What was the CET1 capital ratio?",            "named_metric",  ["CET1"]),
    ("Net Charge-offs trend?",                       "exact_phrase",  ["charge-off", "net charge"]),
    ("What were the main drivers of NII change?",   "analytical",    ["interest income", "NII"]),
    ("How did consumer banking perform?",            "narrative",     ["consumer", "banking"]),
    ("What is the dividend per share?",              "named_metric",  ["dividend"]),
]

def keyword_hit(text, kws):
    t = text.lower()
    return any(kw.lower() in t for kw in kws)

rows = []
for query, qtype, kws in bench_queries:
    for r_name, retriever in [("vector", vec), ("bm25", bm), ("hybrid", hyb)]:
        chunks = retriever.retrieve(query, k=3)
        top1 = chunks[0] if chunks else None
        rows.append({
            "query_type": qtype,
            "query": query[:42],
            "retriever": r_name,
            "top1_page": top1.page_number if top1 else None,
            "kw_hit": keyword_hit(top1.text, kws) if top1 else False,
            "top1_snippet": (top1.text[:75].replace("\\n", " ") + "...") if top1 else "(empty)",
        })
df = pd.DataFrame(rows)
df"""),
        md("""## Step 6: Aggregate - keyword hit rate per (query_type x retriever)

`kw_hit` is a coarse signal (keyword match doesn't equal "actually answered the
query") but **it doesn't lie about direction** - 0% hit rate definitely means
retrieval failed.
"""),
        code("""hit_rate = df.pivot_table(
    values="kw_hit", index="query_type", columns="retriever", aggfunc="mean"
).round(2)

print("Top-1 keyword hit rate by (query_type x retriever):")
print(hit_rate)

# Let the data speak. Don't pre-bake conclusions.
print("\\nWinner per query type:")
for qtype in hit_rate.index:
    winner = hit_rate.loc[qtype].idxmax()
    score = hit_rate.loc[qtype].max()
    print(f"   {qtype:14s} -> {winner} ({score:.0%})")"""),
        md("""**How to read this**:

- **named_metric / exact_phrase**: BM25 usually wins (acronyms, proper nouns,
  hyphenated phrases need exact match)
- **narrative / analytical**: Vector usually wins (semantic alignment)
- **Hybrid**: in theory takes the best of both worlds, but **doesn't always win
  per query** - RRF is unsupervised rank fusion, and on some query distributions
  it dilutes BM25's strong matches

> **If your data shows hybrid losing to a single retriever**: that's not a bug.
> RRF really can underperform "the right retriever" on certain query
> distributions. Production decision: "use whichever retriever wins on our
> actual query log; hybrid is not a free lunch."
"""),
        md("## Step 7: Look inside RRF - how vector and BM25 actually fuse"),
        code("""from src.retrievers.rrf import rrf_merge

q = "What was the CET1 ratio?"
vec_results = vec.search_with_scores(q, k=10)
bm25_results = bm.search_with_scores(q, k=10)

print(f"Query: {q}\\n")
print("Vector top 5:")
for i, (chunk, score) in enumerate(vec_results[:5], 1):
    print(f"  {i}. sim={score:.3f}  p{chunk.page_number}  {chunk.text[:55].strip()}...")

print("\\nBM25 top 5:")
for i, (chunk, score) in enumerate(bm25_results[:5], 1):
    print(f"  {i}. bm25={score:.2f}  p{chunk.page_number}  {chunk.text[:55].strip()}...")

fused = rrf_merge([vec_results, bm25_results], k=60, top_n=5)
print("\\nRRF fused top 5  (RRF score = sum of 1/(k+rank_i) across retrievers):")
for i, (chunk, rrf) in enumerate(fused, 1):
    print(f"  {i}. RRF={rrf:.4f}  p{chunk.page_number}  {chunk.text[:55].strip()}...")"""),
        md("""**Key insight**: RRF **only looks at rank, not at score**. Vector's cosine
distance (0-2 range) and BM25's raw score (any positive number) are on
incompatible scales - directly weighted-summing them would let one side
dominate. RRF uses `1 / (k + rank)` to convert both sides' ranks into
comparable numbers. That's why RRF is called **score-agnostic** fusion.
"""),
        md("## Step 8: Hybrid + parent-child swap (small-to-big retrieval)"),
        code("""from src.chunkers import ParentChildChunker

pc = ParentChildChunker(parent_size=800, child_size=150)
parents, children = pc.chunk_with_parents(docs[0])  # WF only
print(f"{len(parents)} parents, {len(children)} children")

vec.reset(); vec.index(children)
bm = BM25Retriever(); bm.index(children)
parent_store = {p.chunk_id: p for p in parents}
hyb_pc = HybridRetriever(vec, bm, parent_store=parent_store)

q = "What was the CET1 ratio?"
without_pc = hyb_pc.retrieve(q, k=3, use_parent=False)
with_pc    = hyb_pc.retrieve(q, k=3, use_parent=True)

print(f"\\nuse_parent=False: avg {sum(len(c.text) for c in without_pc)//max(1,len(without_pc))} chars/chunk")
print(f"use_parent=True:  avg {sum(len(c.text) for c in with_pc)//max(1,len(with_pc))} chars/chunk")
print("\\nsmall-to-big: child for retrieval (precision), parent for generation (context).")"""),
        md("""## What this notebook actually taught (from your data)

| Theme | What you saw |
|---|---|
| **Cross-doc leakage** | Step 3 exposed the bug; Step 4 fixed it with metadata filter |
| **Metric design** | Don't aggregate `n_pages`; use top-1 keyword hit + manual snippet inspection |
| **Pre-baked vs data-driven conclusions** | We made the mistake in 02; this time we let `hit_rate` speak |
| **RRF is score-agnostic** | Rank, not score - that's what makes vector + BM25 fusable |
| **Small-to-big** | Child for retrieval, parent for LLM context - both retrieval precision and generation context |

## Interview talking point

> "On retrieval benchmarking, my first version used 'unique pages returned' as
> a proxy for retriever quality - because it was cheap. The metric pointed in
> the right direction (hybrid retrieves broader content) but **had no value for
> production decisions**. A `named_metric` query should hit one specific page;
> a `narrative` query should retrieve across pages; **the same metric meant
> opposite things for different query types**.
>
> I switched to per-query top-1 keyword hit + manual snippet inspection. **It's
> not ground truth, just an honest cheap signal.** For real evaluation you go
> to 05_evaluation and use Ragas's `context_precision` / `answer_relevancy`.
>
> **Lesson: cheap proxy metrics are good for fast regression; production
> decisions need semantic-aware metrics.**"

Next, `04_generation.ipynb` connects retrieval to a generation layer with
LangGraph routing.
"""),
    ]
    build(cells, NOTEBOOKS_DIR / "03_retrieval.ipynb")


# =============================================================
# 04_generation.ipynb - LangGraph
# =============================================================
def build_04_generation():
    cells = [
        md("""# 04 - Generation: LangGraph orchestration + Citations

**Aligns with**: S4 Sec. 5 | **Estimated time**: 30 minutes | **Estimated cost**: ~$0.05

## Why LangGraph instead of a plain LCEL chain?

The chain in `00_quickstart` is **linear**: `retriever | prompt | llm | parser`.

But production RAG isn't linear:
- Short Q&A -> top-3 is enough, save tokens
- Long analytical -> top-8 + parent swap
- Empty retrieval -> **don't call the LLM**, refuse directly
- Low confidence -> rewrite query and retry (advanced, left as an exercise)

**This is a state machine with branches** - LangGraph's core use case.

```
[classify_query] -+- factual    -> [quick_retrieve k=3]
                  '- analytical -> [deep_retrieve k=8]
                                     |
                                     +- empty -> [refuse]    -> END
                                     '- found -> [generate]  -> END
```
"""),
        code("""import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()

from src.pipelines.ingestion import IngestionPipeline
from src.pipelines.query import QueryPipeline
from src.chunkers import ParentChildChunker
from src.retrievers import VectorRetriever, BM25Retriever, HybridRetriever
from src.generators import RAGGenerator
from src.observability import CostTracker"""),
        md("## Step 1: A complete ingest + retrieve stack"),
        code("""cost = CostTracker()
pipeline = IngestionPipeline(cost_tracker=cost)
docs = [pipeline.ingest(str(ROOT / f"data/uploads/{n}.pdf")).document
        for n in ["wells_fargo", "tesla", "amd"]]

pc = ParentChildChunker(parent_size=800, child_size=150)
all_parents, all_children = [], []
for doc in docs:
    parents, children = pc.chunk_with_parents(doc)
    all_parents.extend(parents); all_children.extend(children)
print(f"{len(all_parents)} parents, {len(all_children)} children")"""),
        code("""vec = VectorRetriever(
    persist_dir=str(ROOT / "tmp_chroma_04"), collection="lab_04",
    embeddings_cache_dir=ROOT / "cache/embeddings",
)
vec.reset(); vec.index(all_children)
bm = BM25Retriever(); bm.index(all_children)
hybrid = HybridRetriever(vec, bm, parent_store={p.chunk_id: p for p in all_parents})

generator = RAGGenerator(cost_tracker=cost)
qp = QueryPipeline(retriever=hybrid, generator=generator,
                    quick_k=3, deep_k=8, cost_tracker=cost)
print("QueryPipeline (LangGraph) ready")"""),
        md("## Step 2: Inspect the graph - LangGraph has built-in mermaid rendering"),
        code("""print(qp.draw_mermaid())"""),
        code("""try:
    from IPython.display import Image, display
    display(Image(qp.graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG render needs network. Markup is above. {type(e).__name__})")"""),
        md("""## Step 3: Run a few queries, watch the branches

Note the `stages` field - it **records which path through the graph was taken**.
Critical for debugging.
"""),
        code("""queries = [
    "What was Wells Fargo's Q4 2025 net income?",
    "Compare segment performance across the three companies and discuss trends",
    "What was Apple's Q3 revenue?",
]

for q in queries:
    print(f"\\n{'='*70}\\nQ: {q}\\n{'='*70}")
    result = qp.query(q)
    print(f"Stages: {' -> '.join(result['stages'])}")
    print(f"Type:   {result['query_type']}")
    print(f"Answer: {result['answer'][:300]}")
    print(f"Citations: {len(result['citations'])} | n_chunks: {len(result['chunks'])}")
    print(f"Refused: {result['refused']}")"""),
        md("## Step 4: Total cost - accumulated across queries"),
        code("""print("Cost breakdown:")
for stage, c in cost.by_stage.items():
    n = cost.n_calls.get(stage, 0)
    print(f"  {stage:25s} ${c:.5f}  ({n} calls)")
print(f"\\n   TOTAL:    ${cost.total:.5f}")"""),
        md("## Step 5: Citation traceback"),
        code("""result = qp.query("What was Wells Fargo's Q4 2025 net income?")
chunk_lookup = {c.chunk_id: c for c in result["chunks"]}

print(f"Answer: {result['answer']}\\n")
print(f"Cited sources ({len(result['citations'])}):")
for i, cid in enumerate(result["citations"], 1):
    if cid in chunk_lookup:
        ch = chunk_lookup[cid]
        hp = ' > '.join(ch.heading_path) if ch.heading_path else ''
        print(f"\\n  [^{i}] page {ch.page_number} {hp}")
        print(f"      {ch.text[:200]}{'...' if len(ch.text) > 200 else ''}")"""),
        md("""## Recap

| Concept | Implementation |
|---|---|
| State machine | `QueryState` TypedDict passed between nodes |
| Conditional edges | `add_conditional_edges` for routing |
| Refusal short-circuit | empty retrieval -> `refuse` node, no LLM call |
| Observable nodes | every node `@traceable` |
| Cost transparency | shared `CostTracker` |

## Exercise 4.A

Add a **`rewrite_query` node**: when retrieval returns < 2 chunks, ask the LLM
to rewrite the query and retry. Add an `n_rewrites` counter to prevent infinite
loops.
"""),
    ]
    build(cells, NOTEBOOKS_DIR / "04_generation.ipynb")


# =============================================================
# 05_evaluation.ipynb
# =============================================================
def build_05_evaluation():
    cells = [
        md("""# 05 - Evaluation: Baseline vs Improved + Hallucination Detection

**Aligns with**: S4 Sec. 6 | **Estimated time**: 60 minutes | **Estimated cost**: ~$1.00 (full 30 questions x 2 pipelines)

## The question 04 left

We have a working RAG. **How do we prove it's better than quickstart? Better
than the previous version?**

## What this notebook does (around a real production accident)

In an earlier 5-question evaluation, the baseline pipeline answered
"Wells Fargo Q4 2025 net income" as **\\$482M (the actual number is \\$5.4B,
off by a factor of 10)**.

Three things in this notebook:
1. **Run all 30 questions**: Ragas 4 metrics on baseline (RecursiveChunker)
   to quantify failure modes
2. **Fix Q0**: switch to ParentChildChunker (child for retrieval, parent for
   the LLM), rerun, and see whether net income changes from wrong to right
3. **Add Hallucination Detector**: claim-level verification on the wrong Q0
   answer to demonstrate that **the bad \\$482M can be detected automatically**

## The 30-question Golden Set

`data/golden_set/golden.jsonl`:

| Category | Count | What it tests |
|---|---|---|
| fact_finding | 10 | retrieve + extract a number |
| semantic | 8 | semantic alignment |
| single_doc_multihop | 4 | cross-page within one PDF |
| cross_doc | 4 | combine Tesla + AMD |
| out_of_corpus | 4 | refuse behavior |
"""),
        code("""import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()

from src.pipelines.ingestion import IngestionPipeline
from src.pipelines.query import QueryPipeline
from src.chunkers import RecursiveChunker, ParentChildChunker
from src.retrievers import VectorRetriever, BM25Retriever, HybridRetriever
from src.generators import RAGGenerator
from src.evaluators import RagasEvaluator, HallucinationDetector
from src.observability import CostTracker
import pandas as pd, collections, random"""),
        md("## Step 1: Load the golden set"),
        code("""golden = RagasEvaluator.load_golden_set(ROOT / "data/golden_set/golden.jsonl")
print(f"{len(golden)} questions")
for cat, n in collections.Counter(g["category"] for g in golden).most_common():
    print(f"  {cat:25s} {n}")
random.seed(0)
print("\\nSample:")
for g in random.sample(golden, 3):
    print(f"  [{g['category']}] {g['question']}")

# Cost knob: SUBSET vs FULL.
#   SUBSET (5 questions, ~$0.10/pipeline)  -> fast iteration
#   FULL  (30 questions, ~$0.50/pipeline) -> real evaluation
USE_FULL = True   # flip to False for the 5-question fast loop
EVAL_SET = golden if USE_FULL else golden[:5]
print(f"\\nWill evaluate on {len(EVAL_SET)} questions (USE_FULL={USE_FULL})")"""),
        md("""## Step 2: Build the BASELINE pipeline - RecursiveChunker

The "naive default": 400-token chunks, no parent-child. This is the version
that failed Q0 in our earlier 5-question test."""),
        code("""cost_baseline = CostTracker()
pipeline = IngestionPipeline(cost_tracker=cost_baseline)
docs = [pipeline.ingest(str(ROOT / f"data/uploads/{n}.pdf")).document
        for n in ["wells_fargo", "tesla", "amd"]]

# Baseline: recursive chunker, no parent-child
recursive_chunks = []
for d in docs:
    recursive_chunks.extend(RecursiveChunker(chunk_size=400, overlap=60).chunk(d))

vec_b = VectorRetriever(
    persist_dir=str(ROOT / "tmp_chroma_05_baseline"), collection="lab_05_baseline",
    embeddings_cache_dir=ROOT / "cache/embeddings",
)
vec_b.reset(); vec_b.index(recursive_chunks)
bm_b = BM25Retriever(); bm_b.index(recursive_chunks)
hybrid_b = HybridRetriever(vec_b, bm_b)   # no parent_store
qp_baseline = QueryPipeline(hybrid_b, RAGGenerator(cost_tracker=cost_baseline), cost_tracker=cost_baseline)
print(f"baseline pipeline ready ({len(recursive_chunks)} chunks)")"""),
        md("""## Step 3: Run Ragas on baseline

This makes ~150-180 LLM calls (4 metrics x ~30 questions x 1-2 calls each).
At judge-tier model pricing, expect ~$0.40-0.50."""),
        code("""def baseline_query(question: str):
    result = qp_baseline.query(question)
    return {"answer": result["answer"], "chunks": result["chunks"]}

evaluator = RagasEvaluator()
print(f"Evaluating BASELINE on {len(EVAL_SET)} questions...\\n")
baseline_df = evaluator.evaluate(baseline_query, EVAL_SET, verbose=True)

# Save raw results
baseline_df.to_csv(ROOT / "tmp_baseline_ragas.csv", index=False)
print(f"\\nSaved to tmp_baseline_ragas.csv")
print(f"Baseline cost so far: ${cost_baseline.total:.4f}")
baseline_df.head(8)"""),
        md("## Step 4: Score baseline by category - find the weak spots"),
        code("""metric_cols = [c for c in baseline_df.columns if c in
               ("faithfulness","answer_relevancy","context_precision","context_recall")]

if metric_cols and "category" in baseline_df.columns:
    by_cat = baseline_df.groupby("category")[metric_cols].mean().round(3)
    print("Baseline metrics by category:\\n")
    print(by_cat)
    print("\\nThe lowest-scoring category is your highest-leverage improvement target.")
    
    # Find specific failures (faithfulness < 0.5 means likely-wrong answer)
    failures = baseline_df[baseline_df["faithfulness"] < 0.5]
    if len(failures):
        print(f"\\n{len(failures)} likely-wrong-answer cases (faithfulness < 0.5):")
        for _, row in failures.head(3).iterrows():
            print(f"\\n   Q:   {row['user_input'][:80]}")
            print(f"   A:   {str(row['response'])[:120]}")
            print(f"   Ref: {str(row.get('reference', ''))[:120]}")"""),
        md("""## Step 5: Build IMPROVED pipeline - ParentChildChunker + parent expansion

Hypothesis: Q0 failed because the recursive chunker fragmented Wells Fargo's
financial table. Each 400-token chunk had A net-income number, but the wrong
one (a single segment, or Q4 2024). The LLM picked one and ran with it.

The fix: small **child** chunks for retrieval (precision), large **parent**
chunks for the LLM (context including column headers and the section title).

`HybridRetriever` already supports this: pass `parent_store={parent_id: parent_chunk}`
and it auto-swaps children for parents at retrieval time."""),
        code("""cost_improved = CostTracker()

pc = ParentChildChunker(parent_size=800, child_size=150)
all_parents, all_children = [], []
for d in docs:
    parents, children = pc.chunk_with_parents(d)
    all_parents.extend(parents)
    all_children.extend(children)
parent_store = {p.chunk_id: p for p in all_parents}

print(f"  parents: {len(all_parents)}  children: {len(all_children)}")

# Index CHILDREN for retrieval (small, precise), use parent_store at retrieval time
vec_i = VectorRetriever(
    persist_dir=str(ROOT / "tmp_chroma_05_improved"), collection="lab_05_improved",
    embeddings_cache_dir=ROOT / "cache/embeddings",
)
vec_i.reset(); vec_i.index(all_children)
bm_i = BM25Retriever(); bm_i.index(all_children)
hybrid_i = HybridRetriever(vec_i, bm_i, parent_store=parent_store)  # parent expansion ON
qp_improved = QueryPipeline(hybrid_i, RAGGenerator(cost_tracker=cost_improved), cost_tracker=cost_improved)
print(f"improved pipeline ready (retrieval over {len(all_children)} children, generation gets {len(all_parents)} parents)")"""),
        md("## Step 6: Run Ragas on improved pipeline (same questions)"),
        code("""def improved_query(question: str):
    result = qp_improved.query(question)
    return {"answer": result["answer"], "chunks": result["chunks"]}

print(f"Evaluating IMPROVED on {len(EVAL_SET)} questions...\\n")
improved_df = evaluator.evaluate(improved_query, EVAL_SET, verbose=True)
improved_df.to_csv(ROOT / "tmp_improved_ragas.csv", index=False)
print(f"\\nSaved to tmp_improved_ragas.csv")
print(f"Improved cost so far: ${cost_improved.total:.4f}")"""),
        md("## Step 7: Side-by-side comparison"),
        code("""# Aggregate by category, compare
b_by = baseline_df.groupby("category")[metric_cols].mean().round(3)
i_by = improved_df.groupby("category")[metric_cols].mean().round(3)
delta = (i_by - b_by).round(3)

print("BASELINE  (RecursiveChunker, no parent expansion):")
print(b_by)
print("\\nIMPROVED  (ParentChildChunker + parent expansion):")
print(i_by)
print("\\nDELTA  (improved - baseline)  positive = better:")
print(delta)

# Headline: did Q0 get fixed?
q0_b = baseline_df[baseline_df["user_input"].str.contains("Q4 2025 net income", case=False, na=False)]
q0_i = improved_df[improved_df["user_input"].str.contains("Q4 2025 net income", case=False, na=False)]
if len(q0_b) and len(q0_i):
    print("\\n" + "="*70)
    print("Q0 - the production accident from earlier")
    print("="*70)
    print(f"Q: {q0_b.iloc[0]['user_input']}")
    print(f"   Reference: {str(q0_b.iloc[0].get('reference',''))[:100]}")
    print(f"\\n   BASELINE  answer: {str(q0_b.iloc[0]['response'])[:120]}")
    print(f"             faithfulness={q0_b.iloc[0]['faithfulness']:.2f}  context_recall={q0_b.iloc[0]['context_recall']:.2f}")
    print(f"\\n   IMPROVED  answer: {str(q0_i.iloc[0]['response'])[:120]}")
    print(f"             faithfulness={q0_i.iloc[0]['faithfulness']:.2f}  context_recall={q0_i.iloc[0]['context_recall']:.2f}")"""),
        md("""**How to read the results**:
- `delta` is the improvement matrix. `fact_finding` should improve most -
  parent expansion lets the LLM see column headers and the section title,
  preventing pull-the-wrong-number errors
- If a category gets worse, parent_size=800 may be pulling in too much noise.
  In production you tune by query distribution
- Q0 should move from \\$482M toward \\$5.4B. If it's still \\$482M, the
  retriever still didn't find the correct table
"""),
        md("""## Step 8: Hallucination Detector - claim-level validation on Q0

Ragas faithfulness gives one aggregate score. For compliance, you need to know
**which specific claim was wrong**. `HallucinationDetector` decomposes the
answer into atomic claims and verifies each against retrieved context.

This is the safety net for cases where Ragas faithfulness is 0.5
(half-hallucinated): you need to know which half."""),
        code("""detector = HallucinationDetector(cost_tracker=cost_improved)
sample_q = "What was Wells Fargo's Q4 2025 net income?"

# Run on baseline (the one that hallucinated $482M) to demo the detector catching it
baseline_result = qp_baseline.query(sample_q)
print(f"Q: {sample_q}")
print(f"BASELINE answer: {baseline_result['answer']}\\n")

if baseline_result["chunks"]:
    report = detector.detect(baseline_result["answer"], baseline_result["chunks"])
    print(f"Decomposed into {report.n_claims} claims  |  Faithfulness: {report.faithfulness_score:.0%}")
    print(f"   entailed: {report.n_entailed}  unsupported: {report.n_unsupported}  refuted: {report.n_refuted}\\n")
    for cv in report.claims:
        print(f"   [{cv.verdict.upper():12s}] {cv.claim}")
        print(f"      reasoning: {cv.reasoning[:120]}")"""),
        md("## Step 9: OOC refusal validation (compliance)"),
        code("""ooc_qs = [g for g in golden if g["category"] == "out_of_corpus"]
print("Out-of-corpus refusal test (improved pipeline):\\n" + "-"*70)
hits, misses = 0, 0
for g in ooc_qs:
    result = qp_improved.query(g["question"])
    answer = result.get("answer", "").lower()
    refused_correctly = (
        result.get("refused", False)
        or "could not find" in answer
        or "what i found" in answer  # new structured-refusal protocol
        or "what's missing" in answer
    )
    mark = "REFUSED  " if refused_correctly else "ANSWERED!"
    if refused_correctly: hits += 1
    else: misses += 1
    print(f"\\n[{mark}] {g['question']}")
    print(f"   answer: {result['answer'][:160]}")
print(f"\\nOOC refusal rate: {hits}/{hits+misses} = {hits/max(1,hits+misses):.0%}")
print(f"Total cost (baseline + improved + detector): ${cost_baseline.total + cost_improved.total:.4f}")"""),
        md("""## Interview talking point (with real numbers from your run)

> "I built a 30-question golden set across 5 categories (fact_finding,
> semantic, multi-hop, cross-doc, out-of-corpus). On baseline (RecursiveChunker
> 400/60), Ragas faithfulness was X%. **One specific failure** - Wells Fargo
> Q4 2025 net income - was answered as \\$482M instead of \\$5.4B. Diagnosis:
> the recursive chunker fragmented the financial table, and the LLM picked the
> wrong segment number.
>
> I switched to ParentChildChunker (150-token children for retrieval,
> 800-token parents fed to the LLM). The same Q0 then answered correctly.
> Across all 30 questions, faithfulness improved from X% to Y%.
> **A net-income hallucination of \\$482M -> \\$5.4B is the kind of error that
> blocks production deployment in any regulated industry.** I added the
> claim-level HallucinationDetector as a compliance safety net specifically
> for this failure mode."

## Exercise 5.A

Swap the chunker to fixed_size and run the same evaluation. Compare the metric
matrices across all 3 chunkers. **Don't trust "recursive is the best default"
until your own data confirms it.**
"""),
    ]
    build(cells, NOTEBOOKS_DIR / "05_evaluation.ipynb")


# =============================================================
# 06_observability_fastapi.ipynb
# =============================================================
def build_06_observability_fastapi():
    cells = [
        md("""# 06 - Observability + FastAPI: production deployment

**Aligns with**: S4 Sec. 7 | **Estimated time**: 30 minutes | **Estimated cost**: ~$0.05

## What this notebook covers

1. **LangSmith tracing**: every node, every LLM call, every retrieval is
   automatically observable
2. **FastAPI service**: wrap the pipeline as HTTP endpoints - the
   "production-ready" answer interviewers expect
"""),
        md("""## Part 1: LangSmith Tracing

The codebase has `@traceable` decorators throughout
(`src/captioners/vlm_captioner.py`, `src/retrievers/vector.py`,
`src/pipelines/query.py`). As long as `LANGSMITH_API_KEY` is in `.env`, traces
**flow to LangSmith automatically**. No code change required.
"""),
        code("""import sys, warnings, os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()

print(f"LANGSMITH_API_KEY set? {bool(os.getenv('LANGSMITH_API_KEY'))}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT', '(default)')}")"""),
        code("""from src.core.config import configure_langsmith
configure_langsmith()

from src.pipelines.ingestion import IngestionPipeline
from src.pipelines.query import QueryPipeline
from src.chunkers import RecursiveChunker
from src.retrievers import VectorRetriever, BM25Retriever, HybridRetriever
from src.generators import RAGGenerator
from src.observability import CostTracker

cost = CostTracker()
pipeline = IngestionPipeline(cost_tracker=cost)
report = pipeline.ingest(str(ROOT / "data/uploads/wells_fargo.pdf"))
chunks = RecursiveChunker().chunk(report.document)

vec = VectorRetriever(
    persist_dir=str(ROOT / "tmp_chroma_06"), collection="lab_06",
    embeddings_cache_dir=ROOT / "cache/embeddings",
)
vec.reset(); vec.index(chunks)
bm = BM25Retriever(); bm.index(chunks)
hybrid = HybridRetriever(vec, bm)
qp = QueryPipeline(hybrid, RAGGenerator(cost_tracker=cost), cost_tracker=cost)

result = qp.query("What was Wells Fargo's Q4 2025 net income?")
print(f"Total cost: ${cost.total:.5f}")
print(f"Answer: {result['answer'][:200]}")
print("\\nOpen https://smith.langchain.com to see the trace tree.")"""),
        md("""## Trace tree you'll see in LangSmith

```
query_pipeline                                 850ms
  classify_query                                 5ms
  deep_retrieve                                420ms
    hybrid_retrieve                            415ms
      vector_search                            280ms
        OpenAIEmbeddings.embed_query           270ms
      bm25_search                                1ms
  rag_generate                                 420ms
    ChatOpenAI.invoke                          415ms ($0.000012)
```

Click into any node to see input, output, latency, tokens, cost.

**Interview scenario**: "A customer says the RAG answered wrong. How do you debug?"

> Take their query -> search LangSmith -> open the trace -> 99% of the time the
> issue is in retrieval. LangSmith turns debugging from guessing into seeing.
"""),
        md("""## Part 2: FastAPI Service

`src/api/server.py` exposes 3 endpoints:

```
GET  /health    -> status check
POST /ingest    -> upload + index a PDF
POST /query     -> ask a question, get answer + citations
```

Start the server:
```bash
uvicorn src.api.server:app --reload --port 8000
```
"""),
        code("""import httpx
client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=120)

try:
    r = client.get("/health")
    print(f"GET /health -> {r.status_code}")
    print(r.json())
except httpx.ConnectError:
    print("Server not running. Start: uvicorn src.api.server:app --port 8000")"""),
        code("""import json
r = client.post("/ingest", json={"path": str(ROOT / "data/uploads/wells_fargo.pdf")})
print(f"POST /ingest -> {r.status_code}")
print(json.dumps(r.json(), indent=2))"""),
        code("""r = client.post("/query", json={
    "question": "What was Wells Fargo's Q4 2025 net income?", "k": 5,
})
print(f"POST /query -> {r.status_code}\\n")
data = r.json()
print(f"Answer: {data['answer']}\\n")
print(f"Citations ({len(data['citations'])}):")
for c in data["citations"]:
    print(f"  - chunk_id={c['chunk_id'][:16]}, page={c['page_number']}")
    print(f"    {c['text_preview']}")"""),
        md("""## Live curl - shows the service is real

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ingest \\
  -H "Content-Type: application/json" \\
  -d '{"path": "/abs/path/to/wells_fargo.pdf"}'

curl -X POST http://127.0.0.1:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What was Q4 net income?"}'
```
"""),
        md("""## Whole-lab recap - elevator pitch for interviews

> "I built a production RAG service from scratch on three real earnings PDFs:
>
> - **Ingestion**: PyMuPDF + vision LLM captioning of tables/images +
>   3-layer content-addressed cache
> - **Chunking**: 3 strategies, with a `compare_strategies` tool to quantify
>   which fits a corpus
> - **Retrieval**: vector + BM25 + RRF + parent-child swap
> - **Generation**: LangGraph state machine with branching, automatic refusal
>   on empty retrieval
> - **Eval**: Ragas 4 metrics + custom claim-level hallucination detector,
>   30-question golden set across 5 categories
> - **Observability**: every component `@traceable`, full visibility in LangSmith
> - **Service**: FastAPI HTTP wrapper, `POST /ingest` and `POST /query`,
>   curl-able"

Next steps:
- Push the repo to GitHub
- Record a 5-minute demo video (LangSmith trace + curl-ing the service)
- On your resume, write 1-2 lines with numbers ("30-q golden set, faithfulness
  improved from 0.X to 0.Y after adding parent-child + hybrid")
"""),
    ]
    build(cells, NOTEBOOKS_DIR / "06_observability_fastapi.ipynb")


if __name__ == "__main__":
    build_00_quickstart()
    build_01_parsing()
    build_02_chunking()
    build_03_retrieval()
    build_04_generation()
    build_05_evaluation()
    build_06_observability_fastapi()
    print("\nNotebooks built.")
