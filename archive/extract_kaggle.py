"""
filter_and_convert_kaggle.py

Filters the Indian Legal Contract Clauses dataset to license-agreement-relevant
clause types, keeps only High risk rows, annotates them with Groq for
dual-party (Licensor/Licensee) perspective, and outputs a JSONL file
ready to merge with your existing finetune.jsonl.

Usage:
    python filter_and_convert_kaggle.py \
        --input legal_contract_clauses_2.csv \
        --output kaggle_high_risk.jsonl \
        --groq-key YOUR_GROQ_API_KEY

    # To skip Groq and just use the existing risk_level label (faster, less accurate):
    python filter_and_convert_kaggle.py \
        --input legal_contract_clauses_2.csv \
        --output kaggle_high_risk.jsonl \
        --no-reannotate
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORE_LICENSE_TYPES = {
    "License Grant",
    "Ip Ownership Assignment",
    "Non-Transferable License",
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "Irrevocable Or Perpetual License",
    "Unlimited/All-You-Can-Eat-License",
}

RELEVANT_TYPES = {
    "Exclusivity",
    "Anti-Assignment",
    "Cap On Liability",
    "Uncapped Liability",
    "Termination For Convenience",
    "Minimum Commitment",
    "Renewal Term",
    "Change Of Control",
}

ALL_KEEP = CORE_LICENSE_TYPES | RELEVANT_TYPES

SYSTEM_PROMPT = (
    "You are a legal risk analyst specializing in licensing agreements. "
    "Your task is to analyze a contract clause and assess its risk level from a specific party's perspective. "
    "Focus on the practical consequences for the party — what they stand to lose, what obligations they must perform, "
    "and how protected they are if things go wrong. As a rough guide: low risk implies little to no liability "
    "for the party, medium risk implies bounded obligations, and high risk implies significant or uncapped exposure."
)

ANNOTATION_PROMPT = """Analyze the following contract clause from the perspective of the **{party}**.

Clause:
\"\"\"
{clause_text}
\"\"\"

Respond in this exact format:
<scratchpad>
[Your internal reasoning — what obligations, risks, and protections apply to the {party}]
</scratchpad>

Risk Level: [Low/Medium/High]
Explanation: [One or two sentences explaining the risk from the {party}'s perspective]"""

FINETUNE_SYSTEM_PROMPT = (
    "You are a legal risk analyst specializing in licensing agreements. "
    "Your task is to analyze a contract clause and assess its risk level from a specific party's perspective. "
    "Focus on the practical consequences for the party — what they stand to lose, what obligations they must perform, "
    "and how protected they are if things go wrong. As a rough guide: low risk implies little to no liability for the party, "
    "medium risk implies bounded obligations, and high risk implies significant or uncapped exposure."
)

# ---------------------------------------------------------------------------
# Groq annotation
# ---------------------------------------------------------------------------

def call_groq(client, clause_text: str, party: str) -> dict | None:
    """Call Groq API and parse Risk Level + Explanation. Returns None on failure."""
    prompt = ANNOTATION_PROMPT.format(party=party, clause_text=clause_text)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            content = response.choices[0].message.content

            # Strip scratchpad
            content_clean = re.sub(r"<scratchpad>.*?</scratchpad>", "", content, flags=re.DOTALL).strip()

            risk_match = re.search(r"Risk Level:\s*(Low|Medium|High)", content_clean, re.IGNORECASE)
            expl_match = re.search(r"Explanation:\s*(.+)", content_clean, re.DOTALL)

            if not risk_match or not expl_match:
                print(f"  [WARN] Parse failed on attempt {attempt+1}, retrying...")
                time.sleep(1)
                continue

            return {
                "risk": risk_match.group(1).capitalize(),
                "explanation": expl_match.group(1).strip()[:400],
            }

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 30 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {e}")
                time.sleep(2)

    return None


# ---------------------------------------------------------------------------
# JSONL formatting
# ---------------------------------------------------------------------------

def format_finetune_example(clause_text: str, party: str, risk: str, explanation: str, idx: int) -> dict:
    user_content = (
        f"Analyze the following contract clause from the perspective of the **{party}**.\n\n"
        f"Clause:\n\"\"\"\n{clause_text}\n\"\"\"\n\n"
        f"Provide:\n1. Risk Level\n2. Explanation"
    )
    assistant_content = f"Risk Level: {risk}\nExplanation: {explanation}"

    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{FINETUNE_SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_content}"
        "<|eot_id|>"
    )

    return {
        "text": text,
        "_meta": {
            "clause_index": idx,
            "party": party,
            "risk": risk,
            "source": "kaggle_indian_contracts",
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="legal_contract_clauses_2.csv")
    parser.add_argument("--output", default="kaggle_high_risk.jsonl")
    parser.add_argument("--groq-key", default=os.getenv("GROQ_API_KEY"))
    parser.add_argument(
        "--no-reannotate",
        action="store_true",
        help="Skip Groq annotation — use the existing risk_level column directly. "
             "Faster but skips dual-party perspective and explanation generation.",
    )
    parser.add_argument(
        "--high-only",
        action="store_true",
        default=True,
        help="After Groq annotation, only keep rows where both parties are High (default: True). "
             "Set --no-high-only to keep all risk levels.",
    )
    parser.add_argument("--no-high-only", dest="high_only", action="store_false")
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Filter by clause type
    df = df[df["clause_type"].isin(ALL_KEEP)].copy()
    print(f"After clause type filter: {len(df)} rows")

    # Keep only High from source labels (pre-filter before Groq to save API calls)
    df = df[df["risk_level"].str.lower() == "high"].copy()
    print(f"After High-only pre-filter: {len(df)} rows")

    # Deduplicate on clause text
    df = df.drop_duplicates(subset=["clause_text"]).reset_index(drop=True)
    print(f"After deduplication: {len(df)} rows")

    output_path = Path(args.output)

    # --- Mode 1: No reannotation — use source labels directly ---
    if args.no_reannotate:
        print("\nSkipping Groq annotation — using source risk_level directly.")
        print("NOTE: explanations will be empty and party perspective is not applied.")
        written = 0
        with open(output_path, "w") as f:
            for idx, row in df.iterrows():
                for party in ["Licensor", "Licensee"]:
                    ex = format_finetune_example(
                        clause_text=row["clause_text"],
                        party=party,
                        risk="High",
                        explanation=f"This clause poses significant risk to the {party} based on its terms.",
                        idx=idx,
                    )
                    f.write(json.dumps(ex) + "\n")
                    written += 1
        print(f"\nDone. Written {written} examples to {output_path}")
        return

    # --- Mode 2: Reannotate with Groq for dual-party perspectives ---
    try:
        from groq import Groq
    except ImportError:
        print("groq package not found. Run: pip install groq")
        return

    if not args.groq_key:
        print("No Groq API key provided. Use --groq-key or set GROQ_API_KEY env var.")
        return

    client = Groq(api_key=args.groq_key)

    # Resume support — check which indices already done
    done_indices = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_indices.add(rec["_meta"]["clause_index"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_indices)} clause indices already processed")

    written = len(done_indices) * 2  # 2 parties per clause
    skipped = 0

    with open(output_path, "a") as f:
        for idx, row in df.iterrows():
            if idx in done_indices:
                continue

            clause_text = str(row["clause_text"]).strip()
            if len(clause_text) < 30:
                skipped += 1
                continue

            print(f"[{idx+1}/{len(df)}] Annotating clause ({row['clause_type']})...")

            results = {}
            for party in ["Licensor", "Licensee"]:
                result = call_groq(client, clause_text, party)
                if result is None:
                    print(f"  [SKIP] Failed to annotate {party} for clause {idx}")
                    results = {}
                    break
                results[party] = result
                time.sleep(0.3)  # small gap between the two party calls

            if not results:
                skipped += 1
                continue

            # If high_only mode, skip if neither party came back as High
            if args.high_only:
                if not any(r["risk"] == "High" for r in results.values()):
                    print(f"  [SKIP] Neither party rated High after reannotation")
                    skipped += 1
                    continue

            for party, result in results.items():
                ex = format_finetune_example(
                    clause_text=clause_text,
                    party=party,
                    risk=result["risk"],
                    explanation=result["explanation"],
                    idx=idx,
                )
                f.write(json.dumps(ex) + "\n")
                written += 1

            f.flush()

    print(f"\nDone. Written {written} examples, skipped {skipped}.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()