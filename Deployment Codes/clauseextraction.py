"""
clause_extractor.py
====================
Clause segmentation pipeline for CUAD License Agreement PDFs.

Pipeline stages:
  1. PDF -> raw lines  (pdfplumber, with position + font metadata)
  2. Raw lines -> blocks  (gap-threshold + header detection)
  3. Blocks -> clauses    (merge continuations, split sub-clauses, filter boilerplate)
  4. Clauses -> enriched records  (section header, party refs, position, token count)
  5. Output -> JSONL  (one record per clause, ready for annotation)

Each lettered/roman subpart  (a., b. / (a), (b) / i., ii. / (i), (ii))
and each decimal subsection  (2.1, 2.2, 10.3 …)
is emitted as its own JSONL record with parent_clause_id and parent_clause_header
filled in, so an LLM annotator always has the context it needs without receiving
the full parent clause text.

Usage:
    python clause_extractor.py --data_dir ./data --output_dir ./extracted
    python clause_extractor.py --file path/to/agreement.pdf --output_dir ./extracted

Fix history (v2 -> v3):
  BUG 1  _ROMAN pattern could match the empty string; fixed with a mandatory
         positive lookahead (?=[ivxlcdmIVXLCDM]).
  BUG 2  SUBPART_LINE_START required \\s+ between label and content text.
         PDFs routinely produce "(i)\\"Affiliate\\"" with zero spaces; changed
         to \\s* so zero-space cases are caught.
  BUG 3  Decimal subsections (2.1, 2.2, 10.3, "1.1Unless") were not triggering
         new blocks and not recognised as subparts.  Added DECIMAL_SECTION_START
         + _starts_decimal_section(); both build_blocks and merge_and_split_blocks
         now treat decimal-section lines the same as explicit subpart lines.
  BUG 4  merge_and_split_blocks was absorbing a subpart/decimal block into the
         preceding non-subpart block when that block did not end with punctuation.
         Introduced _is_sealed_block(); sealed blocks are emitted immediately
         and never merged forward or backward.
  BUG 5  SUBPART_SPLIT_RE look-behind required \\s+ before the marker, missing
         cases where "(ii)" immediately follows "." with no space.  Changed to
         \\s* (zero-or-more spaces).

Fix history (v3 -> v4):
  BUG 6  Lines beginning with "Section X.Y" (e.g. "Section 1.1.License Grant")
         were not recognised by DECIMAL_SECTION_START because the regex required
         a leading digit.  Added SECTION_PREFIX_DECIMAL pattern and updated
         _starts_decimal_section() to also check for the "Section" prefix.
  BUG 7  TOP_LEVEL_NUMBERED required \\s+ after the clause number's dot, so
         "1.Grant of Rights" (no space) never triggered a new block.  Changed
         to \\s* with a (?!\\d) negative lookahead to avoid matching "1.1".
  BUG 8  Inline splitter SUBPART_SPLIT_RE did not handle "Section X.Y" markers
         appearing after sentence boundaries inside merged text.
  BUG 9  DECIMAL_LABEL_RE could not extract labels from "Section 1.1.Foo";
         added optional (?:sub)?section prefix.
"""

import re
import json
import statistics
import argparse
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    raise SystemExit("pdfplumber not installed. Run: pip install pdfplumber")


# ---------------------------------------------------------------------------
# Constants & tunables
# ---------------------------------------------------------------------------

GAP_MULTIPLIER = 1.5   # gap > GAP_MULTIPLIER × median_line_height  ->  new block

HEADER_PATTERN = re.compile(
    r"^(ARTICLE\s+\w+|SECTION\s+\d+|SCHEDULE\s+\w+|EXHIBIT\s+\w+|ANNEX\s+\w+|"
    r"WHEREAS|NOW[,\s]+THEREFORE|IN WITNESS WHEREOF|RECITALS?|DEFINITIONS?|"
    r"[A-Z][A-Z\s,;]{4,}[A-Z])$"
)

# Top-level numbered clause: "1. …", "2. …", "10. …", "1.Grant …" — NOT "1.1" or "1.1."
# BUG FIX: \s+ changed to \s* so "1.Grant" (zero space) is matched;
# (?!\d) prevents "1.1" from matching as top-level.
TOP_LEVEL_NUMBERED = re.compile(r"^\s*(\d+)\.(?!\d)\s*\S")

# ---------------------------------------------------------------------------
# Roman numeral sub-pattern
# BUG FIX: previous r"(?:x{0,3})(?:ix|iv|v?i{0,3})" matches empty string.
# Fix: mandatory lookahead asserts at least one valid roman-numeral character.
# ---------------------------------------------------------------------------
_ROMAN = r"(?=[ivxlcdmIVXLCDM])(?:x{0,3})(?:ix|iv|v?i{0,3})"

# ---------------------------------------------------------------------------
# Decimal subsection pattern  (2.1 / 2.2 / 10.3 / 1.1.2 / "1.1Unless" …)
# Does NOT match plain "1. " top-level clauses.
# ---------------------------------------------------------------------------
DECIMAL_SECTION_START = re.compile(r"^\s*\d+\.\d+\.?\d*[\s\S]")

# ---------------------------------------------------------------------------
# "Section X.Y" prefix pattern  ("Section 1.1.", "Section 3.2.", "Subsection 2.1")
# Many legal agreements use this format instead of bare decimal numbers.
# ---------------------------------------------------------------------------
SECTION_PREFIX_DECIMAL = re.compile(r"^\s*(?:sub)?section\s+\d+\.\d+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Subpart line opener — all formats:
#   (i)  (ii)  (iii)  (iv)  (v)  (vi) …   roman numerals in parens
#   (a)  (b)  …  (z)                       letters in parens
#   (1)  (2)  …  (99)                      digits in parens
#   i.   ii.  iii.  iv. …                  roman numerals + dot
#   a.   b.   …                            single lowercase letter + dot
#
# BUG FIX: changed terminal \s+\S  ->  \s*\S  so that "(i)"Affiliate"" (zero
# spaces between ")" and the opening quote) is matched correctly.
# ---------------------------------------------------------------------------
SUBPART_LINE_START = re.compile(
    r"^\s*"
    r"(?:"
        r"\((?:" + _ROMAN + r"|[a-zA-Z]|\d{1,2})\)"   # (roman), (letter), or (digit)
        r"|"
        r"(?:" + _ROMAN + r"|[a-z])\."                  # roman. or single-letter.
    r")"
    r"\s*\S",    # \s* — zero or more spaces before the first content character
    re.IGNORECASE,
)

# Merge guard (broader, unchanged from original)
NUMBERED_SECTION = re.compile(
    r"^\s*(\d+\.\d*\.?\s+|[A-Z]\.\s+|\(\w+\)\s+)"
)

# ---------------------------------------------------------------------------
# Inline sub-clause splitter
# Handles: "…end. (b) next…", "…end.(ii)"next"…", "…end. 1.2 next…"
# BUG FIX: look-behind uses \s* (was \s+) so "(ii)" immediately after "." works.
# ---------------------------------------------------------------------------
SUBPART_SPLIT_RE = re.compile(
    r"(?<=[.;])\s*"                                         # sentence boundary + 0+ spaces
    r"("
        r"\((?:" + _ROMAN + r"|[a-zA-Z]|\d{1,2})\)\s*"    # (roman)/(letter)/(digit) + 0+ sp
        r"|"
        r"\d+\.\d+\.?\d*\s+"                               # decimal subsection
        r"|"
        r"(?:sub)?section\s+\d+\.\d+[\. \d]*\.?\s*"        # Section X.Y prefix
    r")",
    re.IGNORECASE,
)

# Extract the leading label from a subpart chunk
SUBPART_LABEL_RE = re.compile(
    r"^\s*"
    r"("
        r"\((?:" + _ROMAN + r"|[a-zA-Z]|\d{1,2})\)"
        r"|"
        r"(?:" + _ROMAN + r"|[a-z])\."
    r")"
    r"\s*",
    re.IGNORECASE,
)

# Extract the leading label from a decimal-section chunk  (e.g. "2.1", "10.3")
DECIMAL_LABEL_RE = re.compile(r"^\s*(?:(?:sub)?section\s+)?(\d+\.\d+\.?\d*)\s*\.?\s*", re.IGNORECASE)

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*[—–]\s*\d+\s*[—–]\s*$"),
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^\s*[ivxlcdmIVXLCDM]{1,6}\s*$"),
    re.compile(r"^(Initialed|Signed|Date|Name|Title|Witnessed By)\s*[:\.]?", re.IGNORECASE),
    re.compile(r"^\s*\[.*?\]\s*$"),
    re.compile(r"^(CONFIDENTIAL|PRIVILEGED AND CONFIDENTIAL|DRAFT)\s*$", re.IGNORECASE),
    re.compile(r"^(IN WITNESS WHEREOF|Signature Page)", re.IGNORECASE),
    re.compile(r"^By:\s*$", re.IGNORECASE),
    re.compile(r"^Exhibit\s+\d+[\.\d]*\s*$", re.IGNORECASE),
    re.compile(r"^(©|Copyright\s+\d{4}|All\s+Rights\s+Reserved)", re.IGNORECASE),
    re.compile(r"^\s*[\w\s\-]+\.pdf\s*$", re.IGNORECASE),
    re.compile(r"^Source\s*:", re.IGNORECASE),
]

PARTY_REF_PATTERN = re.compile(
    r"\b(licensor|licensee|party\s+a|party\s+b|either\s+party|both\s+parties|"
    r"each\s+party|the\s+parties)\b",
    re.IGNORECASE,
)

MIN_CLAUSE_TOKENS = 10


# ---------------------------------------------------------------------------
# Stage 1: PDF -> raw lines
# ---------------------------------------------------------------------------

def extract_raw_lines(pdf_path: str) -> list:
    raw_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            lines = page.extract_text_lines(return_chars=True)
            for line in lines:
                text = line["text"].strip()
                if not text:
                    continue
                chars = line.get("chars", [])
                size  = chars[0].get("size", 8.0) if chars else 8.0
                font  = chars[0].get("fontname", "") if chars else ""
                bold  = "bold" in font.lower() or font.endswith("B") or "BD" in font
                raw_lines.append({
                    "text":   text,
                    "page":   page_num,
                    "top":    line["top"],
                    "bottom": line["bottom"],
                    "height": line["bottom"] - line["top"],
                    "bold":   bold,
                    "size":   size,
                })
    return raw_lines


# ---------------------------------------------------------------------------
# Stage 2: Raw lines -> blocks
# ---------------------------------------------------------------------------

def _is_boilerplate(text: str) -> bool:
    return any(p.match(text) for p in BOILERPLATE_PATTERNS)


def _is_section_header(text: str, bold: bool) -> bool:
    stripped = text.strip()
    if HEADER_PATTERN.match(stripped):
        return True
    if bold and len(stripped) < 65 and stripped == stripped.upper() and 1 <= len(stripped.split()) <= 7:
        return True
    return False


def _starts_new_numbered_clause(text: str) -> bool:
    return bool(TOP_LEVEL_NUMBERED.match(text))


def _starts_decimal_section(text: str) -> bool:
    """
    True for lines like '2.1 …', '2.2 …', '10.3 …', '1.1Unless …',
    as well as 'Section 1.1.License Grant', 'Section 3.2. …', etc.
    Excludes plain '1. …' top-level clauses.
    """
    return (
        (bool(DECIMAL_SECTION_START.match(text)) or bool(SECTION_PREFIX_DECIMAL.match(text)))
        and not _starts_new_numbered_clause(text)
    )


def _starts_subpart(text: str) -> bool:
    """True for (i), (ii), (a), a., i., ii. … — with or without trailing space."""
    return bool(SUBPART_LINE_START.match(text))


def _is_sealed_line(text: str) -> bool:
    """Lines that must always begin a new block and never be absorbed."""
    return (
        _starts_subpart(text)
        or _starts_decimal_section(text)
        or _starts_new_numbered_clause(text)
    )


def _finalize_block(lines: list) -> dict:
    text       = " ".join(l["text"] for l in lines)
    first_text = lines[0]["text"]
    first_bold = lines[0]["bold"]
    is_sub     = _starts_subpart(first_text)
    is_dec     = _starts_decimal_section(first_text)
    return {
        "lines":      lines,
        "text":       text,
        "page":       lines[0]["page"],
        "top":        lines[0]["top"],
        "bottom":     lines[-1]["bottom"],
        "is_header":  _is_section_header(first_text, first_bold),
        "is_subpart": is_sub,
        "is_decimal": is_dec,
        "bold":       all(l["bold"] for l in lines),
    }


def build_blocks(raw_lines: list) -> list:
    """
    Group raw lines into paragraph blocks.

    A new block starts on:
      • page change
      • vertical gap > GAP_MULTIPLIER × median line height
      • current or previous line is a section header
      • current line starts a top-level numbered clause  (1., 2., …)
      • current line starts a decimal subsection         (2.1, 2.2, …)  ← NEW
      • current line starts a lettered/roman subpart     ((i), (a), i., …)

    Continuation lines (wrapped text, explanatory paragraphs like
    "As used in this Agreement…") share no sealed marker so they are absorbed
    into the current block, keeping the full subpart text together.
    """
    if not raw_lines:
        return []

    heights  = [l["height"] for l in raw_lines if l["height"] > 2]
    median_h = statistics.median(heights) if heights else 10.0

    blocks    = []
    cur_lines = [raw_lines[0]]

    for prev, curr in zip(raw_lines, raw_lines[1:]):
        gap     = curr["top"] - prev["bottom"]
        new_block = (
            curr["page"] != prev["page"]
            or gap > GAP_MULTIPLIER * median_h
            or _is_section_header(curr["text"], curr["bold"])
            or _is_section_header(prev["text"], prev["bold"])
            or _is_sealed_line(curr["text"])
        )

        if new_block:
            blocks.append(_finalize_block(cur_lines))
            cur_lines = [curr]
        else:
            cur_lines.append(curr)

    if cur_lines:
        blocks.append(_finalize_block(cur_lines))

    return blocks


# ---------------------------------------------------------------------------
# Stage 3: Merge continuations + split inline sub-clauses
# ---------------------------------------------------------------------------

def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.;:]\s*$", text.rstrip()))


def _is_sealed_block(blk: dict) -> bool:
    """Sealed blocks are never merged forward into another block."""
    return blk["is_subpart"] or blk["is_decimal"]


def merge_and_split_blocks(blocks: list) -> list:
    """
    Pass 1 — Merge continuation blocks.
    Pass 2 — Split inline sub-clause sequences.
    """
    # ── Pass 1 ───────────────────────────────────────────────────────────────
    merged = []
    i = 0
    while i < len(blocks):
        blk = blocks[i]

        if blk["is_header"] or _is_sealed_block(blk):
            merged.append(blk)
            i += 1
            continue

        while (
            i + 1 < len(blocks)
            and not blocks[i + 1]["is_header"]
            and not _is_sealed_block(blocks[i + 1])
            and not _ends_sentence(blk["text"])
            and not NUMBERED_SECTION.match(blocks[i + 1]["text"])
            and not _starts_new_numbered_clause(blocks[i + 1]["text"])
        ):
            nxt = blocks[i + 1]
            blk = {
                "lines":      blk["lines"] + nxt["lines"],
                "text":       blk["text"] + " " + nxt["text"],
                "page":       blk["page"],
                "top":        blk["top"],
                "bottom":     nxt["bottom"],
                "is_header":  False,
                "is_subpart": False,
                "is_decimal": False,
                "bold":       blk["bold"] and nxt["bold"],
            }
            i += 1

        merged.append(blk)
        i += 1

    # ── Pass 2 ───────────────────────────────────────────────────────────────
    final = []
    for blk in merged:
        if blk["is_header"]:
            final.append(blk)
        else:
            final.extend(_split_sub_clauses(blk))

    return final


def _split_sub_clauses(blk: dict) -> list:
    """Split inline (a)…(b)…(c) or (i)…(ii)… or 2.1…2.2 sequences."""
    parts = SUBPART_SPLIT_RE.split(blk["text"])
    if len(parts) <= 1:
        return [blk]

    rebuilt = []
    current = parts[0]
    j = 1
    while j < len(parts):
        marker = parts[j]          if j     < len(parts) else ""
        body   = parts[j + 1]     if j + 1 < len(parts) else ""
        j += 2
        if current.strip():
            rebuilt.append(current.strip())
        current = marker + body
    if current.strip():
        rebuilt.append(current.strip())

    if len(rebuilt) <= 1:
        return [blk]

    result = []
    for part_text in rebuilt:
        is_sub = bool(SUBPART_LABEL_RE.match(part_text))
        is_dec = bool(DECIMAL_LABEL_RE.match(part_text)) and not is_sub
        result.append({
            "lines":      blk["lines"],
            "text":       part_text,
            "page":       blk["page"],
            "top":        blk["top"],
            "bottom":     blk["bottom"],
            "is_header":  False,
            "is_subpart": is_sub,
            "is_decimal": is_dec,
            "bold":       blk["bold"],
        })
    return result


# ---------------------------------------------------------------------------
# Stage 4: Enrich clauses
# ---------------------------------------------------------------------------

def _extract_label(text: str) -> str | None:
    m = SUBPART_LABEL_RE.match(text)
    if m:
        return m.group(1).strip()
    m = DECIMAL_LABEL_RE.match(text)
    return m.group(1).strip() if m else None


def enrich_clauses(blocks: list, doc_total_pages: int) -> list:
    clauses              = []
    current_section      = "PREAMBLE"
    clause_idx           = 0
    last_toplevel_id     = None
    last_toplevel_header = None

    content_blocks = [b for b in blocks if not b["is_header"]]
    content_total  = len(content_blocks)
    content_seen   = 0

    for blk in blocks:
        if blk["is_header"]:
            current_section = blk["text"].strip()
            continue

        if _is_boilerplate(blk["text"]):
            continue

        text   = blk["text"].strip()
        tokens = text.split()
        if len(tokens) < MIN_CLAUSE_TOKENS:
            continue

        content_seen += 1
        rel_pos = content_seen / max(content_total, 1)
        position_label = (
            "early"  if rel_pos < 0.25 else
            "late"   if rel_pos >= 0.75 else
            "middle"
        )

        party_refs = sorted(set(
            m.lower() for m in PARTY_REF_PATTERN.findall(text)
        ))

        inline_match = re.match(r"^([A-Z][A-Z\s]{3,40}):\s+", text)
        inline_label = inline_match.group(1).strip() if inline_match else None

        is_subpart = blk.get("is_subpart", False)
        is_decimal = blk.get("is_decimal", False)
        is_child   = is_subpart or is_decimal

        clause_idx += 1

        clauses.append({
            "clause_id":            clause_idx,
            "text":                 text,
            "section_header":       current_section,
            "inline_label":         inline_label,
            "page":                 blk["page"],
            "position":             position_label,
            "party_refs":           party_refs,
            "token_count":          len(tokens),
            "is_subpart":           is_subpart,
            "is_decimal_section":   is_decimal,
            "subpart_label":        _extract_label(text) if is_child else None,
            "parent_clause_id":     last_toplevel_id     if is_child else None,
            "parent_clause_header": last_toplevel_header if is_child else None,
        })

        if not is_child:
            last_toplevel_id     = clause_idx
            last_toplevel_header = current_section

    return clauses


# ---------------------------------------------------------------------------
# Stage 5: Full pipeline
# ---------------------------------------------------------------------------

def extract_clauses_from_pdf(pdf_path: str) -> dict:
    source = Path(pdf_path).name
    print(f"  [1/4] Extracting raw lines  : {source}")
    raw_lines = extract_raw_lines(str(pdf_path))

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)

    print(f"  [2/4] Building blocks       : {len(raw_lines)} raw lines")
    blocks = build_blocks(raw_lines)

    print(f"  [3/4] Merging / splitting   : {len(blocks)} blocks")
    blocks = merge_and_split_blocks(blocks)

    print(f"  [4/4] Enriching clauses     : {len(blocks)} candidate blocks")
    clauses = enrich_clauses(blocks, total_pages)

    print(f"        => {len(clauses)} clauses extracted.\n")

    return {
        "source_file":  source,
        "total_pages":  total_pages,
        "clause_count": len(clauses),
        "clauses":      clauses,
    }


def _write_txt(clauses: list, txt_path: Path) -> None:
    """Write clause texts as \n\n-separated paragraphs (annotate.py format)."""
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(c["text"] for c in clauses))
    print(f"  TXT  : {txt_path}  ({len(clauses)} clauses)")


def process_directory(data_dir: str, output_dir: str, write_txt: bool = False) -> None:
    data_path   = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{data_dir}'")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{data_dir}'\n{'='*60}")

    all_records  = []
    all_clauses  = []
    summary_rows = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            result = extract_clauses_from_pdf(str(pdf_file))
        except Exception as e:
            print(f"  ERROR processing {pdf_file.name}: {e}\n")
            continue

        out_file = output_path / f"{pdf_file.stem}_clauses.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for clause in result["clauses"]:
                record = {"source_file": result["source_file"], **clause}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_records.append(record)

        all_clauses.extend(result["clauses"])

        if write_txt:
            txt_file = output_path / f"{pdf_file.stem}_clauses.txt"
            _write_txt(result["clauses"], txt_file)

        summary_rows.append({
            "file":    pdf_file.name,
            "pages":   result["total_pages"],
            "clauses": result["clause_count"],
        })

    combined_path = output_path / "all_clauses.jsonl"
    with open(combined_path, "w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if write_txt:
        combined_txt = output_path / "all_clauses.txt"
        _write_txt(all_clauses, combined_txt)

    summary_path = output_path / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "total_pdfs":    len(summary_rows),
            "total_clauses": sum(r["clauses"] for r in summary_rows),
            "per_file":      summary_rows,
        }, fh, indent=2)

    print("=" * 60)
    print(f"DONE")
    print(f"  PDFs processed : {len(summary_rows)}")
    print(f"  Total clauses  : {sum(r['clauses'] for r in summary_rows)}")
    print(f"  Output dir     : {output_path.resolve()}")
    print(f"  all_clauses.jsonl + per-file JSONLs + extraction_summary.json")
    if write_txt:
        print(f"  + per-file .txt files + all_clauses.txt (for annotate.py)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clause segmentation pipeline for License Agreement PDFs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dir", help="Folder containing PDF files")
    group.add_argument("--file",     help="Single PDF to process")
    parser.add_argument(
        "--output_dir", default="./extracted",
        help="Output folder (default: ./extracted)"
    )
    parser.add_argument(
        "--txt", action="store_true",
        help="Also write .txt files (clauses separated by blank lines) for annotate.py"
    )
    args = parser.parse_args()

    if args.file:
        result      = extract_clauses_from_pdf(args.file)
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        out_file    = output_path / f"{Path(args.file).stem}_clauses.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for clause in result["clauses"]:
                record = {"source_file": result["source_file"], **clause}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Written {len(result['clauses'])} clauses -> {out_file}")

        if args.txt:
            txt_file = output_path / f"{Path(args.file).stem}_clauses.txt"
            _write_txt(result["clauses"], txt_file)
    else:
        process_directory(args.data_dir, args.output_dir, write_txt=args.txt)


if __name__ == "__main__":
    main()