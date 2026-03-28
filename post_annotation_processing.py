"""
Legal Clause Dataset Processor
================================
1. Filters out erroneous / low-quality entries from the raw JSONL file
2. Balances the dataset so that Low / Medium / High risk labels occur
   equally across BOTH parties (Licensor + Licensee) in the final file.

Output: processed_dataset.jsonl
"""

import json
import random
from collections import defaultdict, Counter
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "complete_annotation.jsonl"
OUTPUT_FILE = "processed_dataset.jsonl"
REPORT_FILE = "processing_report.txt"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── HELPERS ──────────────────────────────────────────────────────────────────

def load_raw(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append({"_lineno": lineno, **json.loads(line)})
            except json.JSONDecodeError as e:
                print(f"  [!] Skipping line {lineno}: JSON decode error — {e}")
    return records


def filter_errors(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (clean, rejected) lists.
    Only removes records where status == 'api_error'.
    """
    clean, rejected = [], []
    for rec in records:
        if rec.get("status") == "api_error":
            rejected.append(rec)
        else:
            clean.append(rec)
    return clean, rejected


def get_clause_class(rec: dict) -> str:
    """
    Derive a single risk class from both parties using priority rules:
      - If EITHER party is High  → High
      - Else if EITHER party is Medium → Medium
      - Otherwise → Low
    """
    licensor_risk  = rec["parties"]["Licensor"]["risk"]
    licensee_risk  = rec["parties"]["Licensee"]["risk"]
    if "High" in (licensor_risk, licensee_risk):
        return "High"
    if "Medium" in (licensor_risk, licensee_risk):
        return "Medium"
    return "Low"


def get_combined_label(rec: dict) -> str:
    """Combined Licensor|Licensee label string — used only for reporting."""
    parties = rec["parties"]
    return f"{parties['Licensor']['risk']}|{parties['Licensee']['risk']}"


def balance_dataset(clean: list[dict]) -> list[dict]:
    """
    Assign each record a single class via get_clause_class(), then
    downsample every class to the size of the smallest class so that
    Low / Medium / High each make up exactly 33% of the output.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in clean:
        groups[get_clause_class(rec)].append(rec)

    print(f"\n  Clause class distribution before balancing:")
    for cls in ("Low", "Medium", "High"):
        cnt = len(groups.get(cls, []))
        print(f"    {cls:8s}: {cnt:>4d} records")

    present = {k: v for k, v in groups.items() if v}
    target = min(len(v) for v in present.values())
    print(f"\n  Target per class: {target} (size of smallest group)")

    balanced = []
    for cls, recs in present.items():
        sampled = random.sample(recs, target)
        balanced.extend(sampled)
        print(f"    {cls:8s}: kept {len(sampled)} / {len(recs)}")

    random.shuffle(balanced)
    return balanced


def strip_internal_keys(rec: dict) -> dict:
    """Remove processing-only keys before writing output."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(strip_internal_keys(rec), ensure_ascii=False) + "\n")


def write_report(
    raw: list[dict],
    clean: list[dict],
    rejected: list[dict],
    balanced: list[dict],
    path: str,
) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("LEGAL CLAUSE DATASET — PROCESSING REPORT")
    lines.append("=" * 60)
    lines.append(f"\nInput file : {INPUT_FILE}")
    lines.append(f"Output file: {OUTPUT_FILE}")
    lines.append(f"\nTotal raw records  : {len(raw)}")
    lines.append(f"Rejected (errors)  : {len(rejected)}")
    lines.append(f"Clean records      : {len(clean)}")
    lines.append(f"Balanced output    : {len(balanced)}")

    lines.append(f"\n  Records with api_error removed: {len(rejected)}")

    # Pre-balance clause class distribution
    pre_class_counts: Counter = Counter(get_clause_class(r) for r in clean)
    lines.append("\n── Pre-Balance Clause Class Distribution ────────────")
    lines.append("  (High if either party=High, Medium if either=Medium, else Low)")
    for cls in ("Low", "Medium", "High"):
        lines.append(f"  {cls:8s}: {pre_class_counts.get(cls, 0):>4d}")

    # Post-balance clause class distribution
    post_class_counts: Counter = Counter(get_clause_class(r) for r in balanced)
    lines.append("\n── Post-Balance Clause Class Distribution ───────────")
    for cls in ("Low", "Medium", "High"):
        lines.append(f"  {cls:8s}: {post_class_counts.get(cls, 0):>4d}")

    lines.append("\n" + "=" * 60)

    report_text = "\n".join(lines)
    print(report_text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_FILE} …")
    raw = load_raw(INPUT_FILE)
    print(f"  {len(raw)} records loaded")

    print("\nFiltering errors …")
    clean, rejected = filter_errors(raw)
    print(f"  {len(rejected)} rejected  |  {len(clean)} clean")

    print("\nBalancing dataset …")
    balanced = balance_dataset(clean)
    print(f"  {len(balanced)} records in balanced dataset")

    print(f"\nWriting output to {OUTPUT_FILE} …")
    write_jsonl(balanced, OUTPUT_FILE)

    print(f"Writing report to {REPORT_FILE} …")
    write_report(raw, clean, rejected, balanced, REPORT_FILE)

    print("\nDone")


if __name__ == "__main__":
    main()