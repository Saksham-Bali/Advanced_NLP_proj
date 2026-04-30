"""
clause_extractor.py
--------------------
Extracts individual clauses from a contract PDF.

Core heuristic:
    pdfplumber's extract_text() preserves blank lines between paragraphs.
    A blank line (i.e. an empty line in the extracted text) = a clause boundary.
    This directly mirrors how contracts are visually laid out on the page:
    each paragraph separated by a visible gap is treated as one clause.

Usage:
    python clause_extractor.py <path_to_pdf>
"""

import re
import sys
import pdfplumber


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Minimum number of characters for a paragraph to be kept as a clause.
# Filters out noise like page numbers, stray headings, signature lines, etc.
MIN_CLAUSE_LENGTH = 30


# ---------------------------------------------------------------------------
# STEP 1 — Extract full text from PDF, preserving blank lines
# ---------------------------------------------------------------------------

def extract_raw_text(pdf_path: str) -> str:
    """
    Extracts all text from the PDF as a single string.
    pdfplumber preserves line breaks within the text, including blank lines
    between paragraphs — which is exactly what we use to split clauses.

    Pages are joined with a blank line so a paragraph can't accidentally
    straddle a page boundary without a break.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

    # Join pages with a double newline so a clause can't bleed across pages
    return "\n\n".join(pages_text)


# ---------------------------------------------------------------------------
# STEP 2 — Split on blank lines to get raw paragraphs
# ---------------------------------------------------------------------------

def split_into_paragraphs(raw_text: str) -> list[str]:
    """
    Splits the raw text on one or more blank lines.
    Each resulting block is a candidate clause.

    Within each block, wrapped lines (soft line breaks from the PDF renderer)
    are joined back into a single continuous sentence with a space.
    """
    # Split on blank lines (one or more empty lines between content)
    blocks = re.split(r"\n{2,}", raw_text)

    paragraphs = []
    for block in blocks:
        # Join wrapped lines within the paragraph into a single string
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            paragraphs.append(" ".join(lines))

    return paragraphs


# ---------------------------------------------------------------------------
# STEP 3 — Filter noise
# ---------------------------------------------------------------------------

def is_noise(text: str) -> bool:
    """
    Returns True for paragraphs that are almost certainly not substantive clauses:
        - Too short (below MIN_CLAUSE_LENGTH)
        - Pure page numbers  e.g. "- 3 -" or just "3"
        - Signature block fragments  ("By:", "Name:", "Date:", "Its:", "/s/")
        - Source / filing metadata lines (common in SEC filings)
    """
    stripped = text.strip()

    if len(stripped) < MIN_CLAUSE_LENGTH:
        return True

    # Page number patterns
    if re.fullmatch(r"[-–]?\s*\d+\s*[-–]?", stripped):
        return True

    # Signature block lines
    sig_patterns = [
        r"^(By|Name|Its|Title|Date)\s*:",
        r"^/s/",
        r"^_{3,}",
    ]
    for pat in sig_patterns:
        if re.match(pat, stripped, re.IGNORECASE):
            return True

    # Source / filing metadata (common in SEC filings)
    if re.match(r"^Source\s*:", stripped, re.IGNORECASE):
        return True

    return False


def filter_clauses(paragraphs: list[str]) -> list[str]:
    return [p.strip() for p in paragraphs if not is_noise(p.strip())]


# ---------------------------------------------------------------------------
# STEP 4 — Main entry point
# ---------------------------------------------------------------------------

def extract_clauses(pdf_path: str) -> list[str]:
    """
    Full pipeline:  PDF  →  raw text  →  paragraphs  →  filtered clauses

    Returns a list of clause strings.
    """
    raw_text = extract_raw_text(pdf_path)
    paragraphs = split_into_paragraphs(raw_text)
    clauses = filter_clauses(paragraphs)
    return clauses


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clause_extractor.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Output file sits next to the PDF, with _clauses.txt suffix
    # e.g. contracts/agreement.pdf → contracts/agreement_clauses.txt
    base = re.sub(r"\.pdf$", "", pdf_path, flags=re.IGNORECASE)
    output_path = base + "_clauses.txt"

    clauses = extract_clauses(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Extracted {len(clauses)} clause(s) from: {pdf_path}\n")
        f.write("=" * 60 + "\n\n")
        for i, clause in enumerate(clauses, start=1):
            f.write(f"[Clause {i}]\n")
            f.write(clause + "\n\n")

    print(f"Done — {len(clauses)} clauses written to: {output_path}")