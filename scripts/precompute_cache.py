"""
Pre-compute the cache for shipping to students.

Usage:
    python scripts/precompute_cache.py \\
        --inputs data/uploads/wells_fargo.pdf data/uploads/tesla.pdf data/uploads/amd.pdf \\
        --output cache/

This runs the full ingestion pipeline on each PDF (loading, parsing, VLM
captioning) and writes the results into the cache directory. The resulting
cache/ folder can be zipped and shipped to students who unzip it next to
their lab. Then their first `pipeline.ingest("...")` call is a cache hit.

Costs roughly:
    Wells Fargo Q4 2025: ~$0.10 (8 tables, 0 photos)
    Tesla Q1 2026:       ~$0.40 (10 tables, 7 photos)
    AMD Q4 2025:         ~$0.30 (15 tables, 5 charts)
    -------------
    Total:               ~$0.80 (one-time, you pay)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running from repo root: `python scripts/precompute_cache.py ...`
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipelines.ingestion import IngestionPipeline
from src.core.cache import CacheBundle


def main():
    parser = argparse.ArgumentParser(description="Pre-compute ingestion cache")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more PDF files to ingest",
    )
    parser.add_argument(
        "--output", default="cache",
        help="Cache output directory (default: cache/)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Limit pages per document (for testing)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear existing cache first",
    )
    args = parser.parse_args()
    
    # Pre-flight: verify all inputs exist BEFORE doing any work.
    # This prevents the silent "warning + zip an empty cache" failure mode.
    missing = [p for p in args.inputs if not Path(p).exists()]
    if missing:
        print("The following input files do not exist:")
        for p in missing:
            print(f"     {p}")
        print()
        print(" Did you download the PDFs into data/uploads/ first?")
        print(" See LAB_HANDOUT.md footnote 1 for source URLs.")
        sys.exit(1)
    
    cache = CacheBundle.from_root(args.output, enabled=True)
    pipeline = IngestionPipeline(cache=cache)
    
    if args.clear:
        cleared = pipeline.clear_cache()
        print(f"Cleared cache: {cleared}")
    
    print(f"Pre-computing cache -> {args.output}/")
    print(f" Captioner: {pipeline.captioner.name}")
    print(f" Inputs: {len(args.inputs)} PDF(s)")
    print()
    
    total_cost = 0.0
    for pdf_path in args.inputs:
        pdf_path = Path(pdf_path)
        print(f"{pdf_path.name}")
        report = pipeline.ingest(pdf_path, max_pages=args.max_pages, verbose=True)
        print(report.summary())
        total_cost += report.total_cost_usd
        print()
    
    print(f"Grand total: ${total_cost:.4f}")
    print(f"Cache ready at: {args.output}/")
    print(f" To ship to students: zip -r cache_bundle.zip {args.output}/")


if __name__ == "__main__":
    main()
