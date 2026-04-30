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

Usage:
    # Process entire data folder
    python clause_extractor.py --data_dir ./data --output_dir ./extracted

    # Single file
    python clause_extractor.py --file path/to/agreement.pdf --output_dir ./extracted
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

GAP_MULTIPLIER = 1.5   # gap > GAP_MULTIPLIER x median_line_height -> new block

HEADER_PATTERN = re.compile(
    r"^(ARTICLE\s+\w+|SECTION\s+\d+|SCHEDULE\s+\w+|EXHIBIT\s+\w+|ANNEX\s+\w+|"
    r"WHEREAS|NOW[,\s]+THEREFORE|IN WITNESS WHEREOF|RECITALS?|DEFINITIONS?|"
    r"[A-Z][A-Z\s,;]{4,}[A-Z])$"
)

NUMBERED_SECTION = re.compile(
    r"^\s*(\d+\.\d*\.?\s+|[A-Z]\.\s+|\(\w+\)\s+)"
)

# Matches the START of a top-level numbered clause: "1.", "2.", "10." etc.
# Intentionally does NOT match sub-clauses like "1.1" or "(a)" — those are
# handled separately inside _split_sub_clauses.
TOP_LEVEL_NUMBERED = re.compile(r"^\s*(\d+)\.\s+\S")

BOILERPLATE_PATTERNS = [
    # Page numbers: "Page 3", "Page 3 of 12", "- 3 -", "— 3 —"
    re.compile(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*[—–]\s*\d+\s*[—–]\s*$"),
    # Bare standalone integers (lone page numbers with nothing else on the line)
    re.compile(r"^\s*\d{1,4}\s*$"),
    # Roman numeral page numbers (i, ii, iii, iv ...)
    re.compile(r"^\s*[ivxlcdmIVXLCDM]{1,6}\s*$"),
    # Signature / form fields
    re.compile(r"^(Source|Initialed|Signed|Date|Name|Title|Witnessed By)\s*[:\.]?", re.IGNORECASE),
    re.compile(r"^\s*\[.*?\]\s*$"),
    # Document status stamps
    re.compile(r"^(CONFIDENTIAL|PRIVILEGED AND CONFIDENTIAL|DRAFT)\s*$", re.IGNORECASE),
    re.compile(r"^(IN WITNESS WHEREOF|Signature Page)", re.IGNORECASE),
    re.compile(r"^By:\s*$", re.IGNORECASE),
    re.compile(r"^Exhibit\s+\d+[\.\d]*\s*$", re.IGNORECASE),
    # Copyright / filename footers
    re.compile(r"^(©|Copyright\s+\d{4}|All\s+Rights\s+Reserved)", re.IGNORECASE),
    re.compile(r"^\s*[\w\s\-]+\.pdf\s*$", re.IGNORECASE),
    # SEC filing source attribution: "Source: COMPANY NAME, 8-K, 8/15/2019"
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
    """
    Extract all lines with position and font metadata.
    Returns list of dicts: {text, page, top, bottom, height, bold, size}
    """
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
    # Bold, short, all-caps lines are section headers
    if bold and len(stripped) < 65 and stripped == stripped.upper() and 1 <= len(stripped.split()) <= 7:
        return True
    return False


def _starts_new_numbered_clause(text: str) -> bool:
    """
    Returns True if this line opens a top-level numbered clause like "1. ...",
    "2. ...", "10. ..." — but NOT sub-clauses like "1.1 ..." or "(a) ...".
    This keeps numbered items as individual clauses rather than merging them
    together, while leaving sub-clause splitting to _split_sub_clauses.
    """
    return bool(TOP_LEVEL_NUMBERED.match(text))


def _finalize_block(lines: list) -> dict:
    text = " ".join(l["text"] for l in lines)
    bold = all(l["bold"] for l in lines)
    return {
        "lines":     lines,
        "text":      text,
        "page":      lines[0]["page"],
        "top":       lines[0]["top"],
        "bottom":    lines[-1]["bottom"],
        "is_header": _is_section_header(lines[0]["text"], lines[0]["bold"]),
        "bold":      bold,
    }


def build_blocks(raw_lines: list) -> list:
    """
    Group raw lines into paragraph blocks.
    New block starts when:
      - page changes
      - vertical gap > GAP_MULTIPLIER x median line height
      - current or previous line is a section header
      - current line starts a top-level numbered clause (1., 2., 3. ...)
    """
    if not raw_lines:
        return []

    heights  = [l["height"] for l in raw_lines if l["height"] > 2]
    median_h = statistics.median(heights) if heights else 10.0

    blocks    = []
    cur_lines = [raw_lines[0]]

    for prev, curr in zip(raw_lines, raw_lines[1:]):
        gap         = curr["top"] - prev["bottom"]
        page_change = curr["page"] != prev["page"]
        curr_hdr    = _is_section_header(curr["text"], curr["bold"])
        prev_hdr    = _is_section_header(prev["text"], prev["bold"])
        curr_numbered = _starts_new_numbered_clause(curr["text"])

        new_block = (
            page_change
            or gap > GAP_MULTIPLIER * median_h
            or curr_hdr
            or prev_hdr
            or curr_numbered          # <-- each "N. ..." starts its own block
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
# Stage 3: Blocks -> clauses (merge continuations + split sub-clauses)
# ---------------------------------------------------------------------------

def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.;:]\s*$", text.rstrip()))


def merge_and_split_blocks(blocks: list) -> list:
    """
    Pass 1 — Merge continuation blocks:
      A non-header block that doesn't end a sentence AND the following block
      doesn't start with a numbered clause or sub-clause gets merged forward.
      Top-level numbered clauses (1., 2., 3. ...) are NEVER merged into their
      predecessor — each is preserved as a standalone clause.

    Pass 2 — Split numbered sub-clause sequences:
      If a block contains (a)...(b)...(c) or 1.1...1.2... markers
      at sentence boundaries, split into individual sub-clause blocks.
    """
    # --- Pass 1: merge continuations ---
    merged = []
    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if blk["is_header"]:
            merged.append(blk)
            i += 1
            continue

        # Never merge forward if the next block is a top-level numbered clause
        while (
            i + 1 < len(blocks)
            and not blocks[i + 1]["is_header"]
            and not _ends_sentence(blk["text"])
            and not NUMBERED_SECTION.match(blocks[i + 1]["text"])
            and not _starts_new_numbered_clause(blocks[i + 1]["text"])
        ):
            nxt = blocks[i + 1]
            blk = {
                "lines":     blk["lines"] + nxt["lines"],
                "text":      blk["text"] + " " + nxt["text"],
                "page":      blk["page"],
                "top":       blk["top"],
                "bottom":    nxt["bottom"],
                "is_header": False,
                "bold":      blk["bold"] and nxt["bold"],
            }
            i += 1

        merged.append(blk)
        i += 1

    # --- Pass 2: split sub-clause sequences ---
    final = []
    for blk in merged:
        if blk["is_header"]:
            final.append(blk)
            continue
        final.extend(_split_sub_clauses(blk))

    return final


def _split_sub_clauses(blk: dict) -> list:
    """
    Split at points like: '...sentence end. (b) Next clause...'
    or '...sentence end. 1.2 Next clause...'
    Keeps opening preamble with the first sub-clause.
    Does NOT split on top-level numbered patterns (1., 2., ...) because those
    are already separated at the block-building stage.
    """
    text = blk["text"]
    split_re = re.compile(
        r"(?<=[.;])\s+(\([a-zA-Z0-9ivxlcdm]{1,4}\)\s+|\d+\.\d+\.?\s+)"
    )
    parts = split_re.split(text)

    if len(parts) <= 1:
        return [blk]

    rebuilt = []
    current = parts[0]
    j = 1
    while j < len(parts):
        marker = parts[j] if j < len(parts) else ""
        body   = parts[j + 1] if j + 1 < len(parts) else ""
        j += 2
        if current.strip():
            rebuilt.append(current.strip())
        current = marker + body

    if current.strip():
        rebuilt.append(current.strip())

    if len(rebuilt) <= 1:
        return [blk]

    return [
        {
            "lines":     blk["lines"],
            "text":      part_text,
            "page":      blk["page"],
            "top":       blk["top"],
            "bottom":    blk["bottom"],
            "is_header": False,
            "bold":      blk["bold"],
        }
        for part_text in rebuilt
    ]


# ---------------------------------------------------------------------------
# Stage 4: Enrich clauses with metadata
# ---------------------------------------------------------------------------

def enrich_clauses(blocks: list, doc_total_pages: int) -> list:
    """
    Walk blocks tracking section context. For each substantive block,
    emit a clause record with metadata useful for annotation and fine-tuning.
    """
    clauses         = []
    current_section = "PREAMBLE"
    clause_idx      = 0
    content_blocks  = [b for b in blocks if not b["is_header"]]
    content_total   = len(content_blocks)
    content_seen    = 0

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
        if rel_pos < 0.25:
            position_label = "early"
        elif rel_pos < 0.75:
            position_label = "middle"
        else:
            position_label = "late"

        party_refs = sorted(set(
            m.lower() for m in PARTY_REF_PATTERN.findall(text)
        ))

        # Detect inline sub-heading label (e.g. "TERMINATION FOR CAUSE: ...")
        inline_match = re.match(r"^([A-Z][A-Z\s]{3,40}):\s+", text)
        inline_label = inline_match.group(1).strip() if inline_match else None

        clause_idx += 1
        clauses.append({
            "clause_id":      clause_idx,
            "text":           text,
            "section_header": current_section,
            "inline_label":   inline_label,
            "page":           blk["page"],
            "position":       position_label,
            "party_refs":     party_refs,
            "token_count":    len(tokens),
        })

    return clauses


# ---------------------------------------------------------------------------
# Stage 5: Full pipeline
# ---------------------------------------------------------------------------

def extract_clauses_from_pdf(pdf_path: str) -> dict:
    """
    Run the full pipeline on a single PDF.
    Returns {source_file, total_pages, clause_count, clauses}.
    """
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


def process_directory(data_dir: str, output_dir: str) -> None:
    data_path   = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{data_dir}'")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{data_dir}'\n{'='*60}")

    all_records  = []
    summary_rows = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            result = extract_clauses_from_pdf(str(pdf_file))
        except Exception as e:
            print(f"  ERROR processing {pdf_file.name}: {e}\n")
            continue

        # Per-file JSONL
        out_file = output_path / f"{pdf_file.stem}_clauses.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for clause in result["clauses"]:
                record = {"source_file": result["source_file"], **clause}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_records.append(record)

        summary_rows.append({
            "file":    pdf_file.name,
            "pages":   result["total_pages"],
            "clauses": result["clause_count"],
        })

    # Combined JSONL
    combined_path = output_path / "all_clauses.jsonl"
    with open(combined_path, "w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary JSON
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
    else:
        process_directory(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()