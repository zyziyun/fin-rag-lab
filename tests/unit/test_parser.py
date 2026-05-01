"""Tests for PDFStructuralParser helpers (no I/O, no PDF needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parsers.pdf_parser import PDFStructuralParser


# ---- _reflow_spaced_caps: regression for Tesla-style "S U M M A R Y" PDFs
class TestReflowSpacedCaps:
    def test_collapses_simple_run(self):
        # Tesla's SUMMARY HIGHLIGHTS section header
        # _reflow_spaced_caps preserves multi-space gaps (those are word boundaries);
        # the final \s+ collapse happens in _clean_text downstream.
        out = PDFStructuralParser._reflow_spaced_caps("S U M M A R Y  H I G H L I G H T S")
        assert out == "SUMMARY  HIGHLIGHTS"

    def test_collapses_long_word(self):
        out = PDFStructuralParser._reflow_spaced_caps("S U P P O R T I N G  I N F R A S T R U C T U R E")
        assert out == "SUPPORTING  INFRASTRUCTURE"

    def test_handles_digit_in_run(self):
        # Tesla has "C O R T E X 2"
        out = PDFStructuralParser._reflow_spaced_caps("C O R T E X 2 - B U I L D I N G")
        assert out == "CORTEX2 - BUILDING"

    def test_does_not_break_normal_text(self):
        # Sentences with regular words must pass through unchanged
        text = "Wells Fargo Q4 2025 results were strong this quarter."
        assert PDFStructuralParser._reflow_spaced_caps(text) == text

    def test_does_not_collapse_short_acronyms(self):
        # Two-letter sequences are below the 3-token threshold; leave alone
        # (avoids false positives like 'A B' or 'P R')
        assert PDFStructuralParser._reflow_spaced_caps("A B") == "A B"

    def test_does_not_touch_lowercase(self):
        assert PDFStructuralParser._reflow_spaced_caps("a b c d") == "a b c d"

    def test_preserves_existing_words(self):
        assert PDFStructuralParser._reflow_spaced_caps("CET1 ratio was 11.7%") == "CET1 ratio was 11.7%"

    def test_collapses_through_full_clean_text_pipeline(self):
        # End-to-end via _clean_text: order matters (reflow before \s+ collapse)
        parser = PDFStructuralParser()
        out = parser._clean_text("  S U M M A R Y  H I G H L I G H T S  ")
        assert out == "SUMMARY HIGHLIGHTS"
