"""Generate a synthetic financial-style PDF for testing the ingestion pipeline.

This PDF mimics the structure of a real earnings report:
  - A title heading
  - Paragraphs of prose
  - A markdown-renderable table
  - A simple image (a colored rectangle, just to test image extraction)

Used by test_ingestion.py and as a sample for offline development.
"""
from pathlib import Path
import fitz  # PyMuPDF


def make_test_pdf(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open()
    
    # ---- Page 1 ----
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 72),
        "ACME CORP Q4 2025 EARNINGS REPORT",
        fontsize=18,
        fontname="helv",
    )
    page.insert_text(
        (72, 110),
        "Highlights",
        fontsize=14,
        fontname="helv",
    )
    page.insert_text(
        (72, 140),
        "ACME Corp reported Q4 2025 net income of $5.4 billion, up 6% year-over-year.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (72, 160),
        "Diluted earnings per share were $1.62 compared to $1.43 in Q4 2024.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (72, 200),
        "Financial Summary",
        fontsize=14,
        fontname="helv",
    )
    
    # Insert a simple table by drawing it as text rows
    table_y_start = 230
    rows = [
        ["Metric", "Q4 2024", "Q4 2025"],
        ["Revenue ($B)", "20.4", "21.3"],
        ["Net Income ($B)", "5.1", "5.4"],
        ["EPS ($)", "1.43", "1.62"],
        ["ROE (%)", "11.7", "12.3"],
    ]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            page.insert_text((72 + j * 120, table_y_start + i * 20), cell, fontsize=10)
    
    # ---- Page 2 ----
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text(
        (72, 72),
        "Operating Segment Performance",
        fontsize=14,
        fontname="helv",
    )
    page2.insert_text(
        (72, 110),
        "Consumer Banking and Lending",
        fontsize=12,
        fontname="helv",
    )
    page2.insert_text(
        (72, 140),
        "Revenue increased 7% YoY to $9.6 billion, driven by lower deposit pricing,",
        fontsize=11,
        fontname="helv",
    )
    page2.insert_text(
        (72, 160),
        "higher deposit and loan balances, and improved credit card spending.",
        fontsize=11,
        fontname="helv",
    )
    page2.insert_text(
        (72, 200),
        "Commercial Banking",
        fontsize=12,
        fontname="helv",
    )
    page2.insert_text(
        (72, 230),
        "Revenue decreased 3% YoY due to lower interest rates, partially offset by",
        fontsize=11,
        fontname="helv",
    )
    page2.insert_text(
        (72, 250),
        "higher loan balances and lower deposit pricing.",
        fontsize=11,
        fontname="helv",
    )
    
    # Insert a small "chart-like" image (just a rectangle)
    rect = fitz.Rect(72, 320, 300, 450)
    page2.draw_rect(rect, fill=(0.3, 0.5, 0.8), color=(0, 0, 0))
    page2.insert_text((100, 470), "[Chart: Revenue by Segment]", fontsize=10)
    
    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    out = make_test_pdf("data/uploads/synthetic_earnings.pdf")
    print(f"Generated: {out}")
