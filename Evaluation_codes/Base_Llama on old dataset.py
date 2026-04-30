"""
baseline_eval.py
────────────────
Zero-shot Llama 8B (via Ollama) baseline evaluation pipeline
for contract clause risk classification.

Evaluates both Licensor and Licensee perspectives separately
and produces a full classification report for each.

Usage:
    python baseline_eval.py --input annotated_clauses.jsonl
"""

import argparse
import json
import re
import time
import requests
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
LABELS       = ["low", "medium", "high"]
RESULTS_DIR  = Path("baseline_results2")

# ── 1. DATA LOADER ────────────────────────────────────────────────────────────

def load_annotated_data(filepath: str) -> tuple[list[dict], list[dict]]:
    """
    Parse NDJSON annotation file (one JSON object per line).

    Expected schema per line:
    {
        "clause_text": "...",
        "clause_index": 6,
        "status": "success",
        "parties": {
            "Licensor": {"risk": "Low", "explanation": "..."},
            "Licensee": {"risk": "High", "explanation": "..."}
        }
    }

    Returns:
        licensor_records, licensee_records  — each a list of dicts with:
            clause_text, clause_index, risk_level, gt_explanation
    """
    licensor_records = []
    licensee_records = []

    raw = Path(filepath).read_text(encoding="utf-8").strip()

    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  [skip] line {line_no} — bad JSON: {e}")
            continue

        if obj.get("status") != "success":
            continue

        clause_text  = obj.get("clause_text", "").strip()
        clause_index = obj.get("clause_index", -1)
        parties      = obj.get("parties", {})

        if not clause_text:
            continue

        if "Licensor" in parties:
            licensor_records.append({
                "clause_text":    clause_text,
                "clause_index":   clause_index,
                "risk_level":     parties["Licensor"]["risk"].lower(),
                "gt_explanation": parties["Licensor"].get("explanation", ""),
            })

        if "Licensee" in parties:
            licensee_records.append({
                "clause_text":    clause_text,
                "clause_index":   clause_index,
                "risk_level":     parties["Licensee"]["risk"].lower(),
                "gt_explanation": parties["Licensee"].get("explanation", ""),
            })

    return licensor_records, licensee_records


# ── 2. PROMPT BUILDER ─────────────────────────────────────────────────────────

def build_prompt(clause_text: str, party: str) -> str:
    return f"""You are a contract risk analyst reviewing a clause from a commercial agreement.

Your task: classify the risk level of this clause FROM THE PERSPECTIVE OF THE {party.upper()}.

Respond with ONLY one of these three words — nothing else:
Low
Medium
High

Clause:
\"\"\"{clause_text.strip()}\"\"\"

Risk level for {party}:"""


# ── 3. OLLAMA INFERENCE ───────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,    # greedy decoding — reproducible baseline
            "num_predict": 16,   # we only need one word back
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ── 4. RESPONSE PARSER ────────────────────────────────────────────────────────

def parse_label(raw: str) -> str | None:
    """
    Extract low / medium / high from raw model output.
    Checks high first to avoid 'highlight' matching as 'high'.
    Returns None if no label found (caller handles the fallback).
    """
    text = raw.lower()
    for label in ["high", "medium", "low"]:
        if re.search(rf"\b{label}\b", text):
            return label
    return None


# ── 5. EVALUATION LOOP ────────────────────────────────────────────────────────

def evaluate_party(records: list[dict], party: str) -> dict:
    """
    Run zero-shot inference on every clause for one party perspective.

    Returns a results dict containing:
        party, predictions (list), parse_failures (list)
    """
    predictions    = []
    parse_failures = []

    print(f"\n── {party} perspective  ({len(records)} clauses) ──────────────")

    for i, item in enumerate(records):
        clause = item["clause_text"]
        gt     = item["risk_level"]

        prompt = build_prompt(clause, party)

        try:
            raw  = call_ollama(prompt)
            pred = parse_label(raw)
        except requests.RequestException as e:
            print(f"  [error] clause {item['clause_index']}: {e}")
            raw, pred = "", None

        if pred is None:
            parse_failures.append({
                "clause_index": item["clause_index"],
                "raw_output":   raw,
                "clause":       clause[:120],
            })
            pred = "low"    # fallback — penalises the model fairly

        correct = pred == gt
        status  = "✓" if correct else "✗"

        print(
            f"  [{i+1:>3}/{len(records)}] "
            f"clause {item['clause_index']:>3}  "
            f"gt={gt:<7} pred={pred:<7} {status}"
        )

        predictions.append({
            "clause_index":   item["clause_index"],
            "clause":         clause[:300],
            "ground_truth":   gt,
            "predicted":      pred,
            "raw_output":     raw,
            "correct":        correct,
            "gt_explanation": item.get("gt_explanation", ""),
        })

    return {
        "party":          party,
        "predictions":    predictions,
        "parse_failures": parse_failures,
    }


# ── 6. REPORT GENERATOR ───────────────────────────────────────────────────────

def build_report(results: dict) -> str:
    party       = results["party"]
    predictions = results["predictions"]
    failures    = results["parse_failures"]

    y_true = [r["ground_truth"] for r in predictions]
    y_pred = [r["predicted"]    for r in predictions]

    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    cls_report = classification_report(
        y_true, y_pred,
        labels=LABELS,
        target_names=[l.capitalize() for l in LABELS],
        digits=3,
        zero_division=0,
    )

    cm    = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index   =[f"true_{l}"  for l in LABELS],
        columns =[f"pred_{l}"  for l in LABELS],
    )

    # misclassification frequency table
    errors = [r for r in predictions if not r["correct"]]
    if errors:
        err_df = (
            pd.DataFrame(errors)[["ground_truth", "predicted"]]
            .value_counts()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        err_str = err_df.to_string(index=False)
    else:
        err_str = "  none — perfect score"

    # high <-> low flips are the most dangerous errors for end users
    flip_errors = [
        r for r in errors
        if {r["ground_truth"], r["predicted"]} == {"high", "low"}
    ]

    lines = [
        "=" * 64,
        f"  BASELINE  ·  {party.upper()} PERSPECTIVE  ·  Llama 8B  (zero-shot)",
        "=" * 64,
        "",
        f"  Model               : {OLLAMA_MODEL}",
        f"  Clauses evaluated   : {len(predictions)}",
        f"  Parse failures      : {len(failures)}  (defaulted to 'low')",
        "",
        f"  Overall accuracy    : {acc:.3f}",
        f"  Balanced accuracy   : {bal_acc:.3f}  (macro-averaged recall)",
        "",
        "  Note: if balanced accuracy << overall accuracy, the model is",
        "  leaning on the majority class rather than learning risk signals.",
        "",
        "── Per-class metrics ───────────────────────────────────────────",
        "",
        cls_report,
        "── Confusion matrix ────────────────────────────────────────────",
        "",
        cm_df.to_string(),
        "",
        "  Rows = ground truth   Columns = model prediction",
        "",
        "── Misclassification breakdown ─────────────────────────────────",
        "",
        err_str,
        "",
        f"── High ↔ Low flip errors  ({len(flip_errors)} found) ─────────────────────",
        "  These are the highest-priority errors — model called a High-risk",
        "  clause Low (or vice versa). Prioritise these in fine-tuning.",
        "",
    ]

    if flip_errors:
        for r in flip_errors[:5]:
            lines += [
                f"  clause {r['clause_index']:>3}  gt={r['ground_truth']}  "
                f"pred={r['predicted']}",
                f"  text   : {r['clause'][:120]}...",
                f"  reason : {r['gt_explanation'][:140]}",
                "",
            ]
        if len(flip_errors) > 5:
            lines.append(f"  ... and {len(flip_errors) - 5} more (see predictions JSON)")
    else:
        lines.append("  none")

    lines += ["", "=" * 64]
    return "\n".join(lines)


# ── 7. ENTRY POINT ────────────────────────────────────────────────────────────

def main():
    global OLLAMA_MODEL
    parser = argparse.ArgumentParser(description="Zero-shot Llama baseline for contract risk classification")
    parser.add_argument("--input",  default="annotated_clauses.jsonl", help="Path to NDJSON annotation file")
    parser.add_argument("--model",  default=OLLAMA_MODEL,              help="Ollama model tag")
    parser.add_argument("--outdir", default=str(RESULTS_DIR),          help="Directory to write results into")
    args = parser.parse_args()

    OLLAMA_MODEL = args.model

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"Loading annotations from {args.input} ...")
    licensor_records, licensee_records = load_annotated_data(args.input)
    print(f"  Licensor clauses : {len(licensor_records)}")
    print(f"  Licensee clauses : {len(licensee_records)}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    t0 = time.time()

    licensor_results = evaluate_party(licensor_records, "Licensor")
    licensee_results = evaluate_party(licensee_records, "Licensee")

    elapsed = time.time() - t0
    print(f"\nInference complete in {elapsed:.1f}s")

    # ── save raw results ──────────────────────────────────────────────────────
    for res in [licensor_results, licensee_results]:
        party = res["party"].lower()

        pred_path = out / f"{party}_predictions.json"
        pred_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Predictions saved → {pred_path}")

        if res["parse_failures"]:
            pf_path = out / f"{party}_parse_failures.json"
            pf_path.write_text(json.dumps(res["parse_failures"], indent=2), encoding="utf-8")
            print(f"  ⚠  {len(res['parse_failures'])} parse failures → {pf_path}")

    # ── build and print reports ───────────────────────────────────────────────
    for res in [licensor_results, licensee_results]:
        report = build_report(res)
        print("\n" + report)

        report_path = out / f"{res['party'].lower()}_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()