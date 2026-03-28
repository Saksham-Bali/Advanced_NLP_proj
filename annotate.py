"""
Legal Clause Risk Annotation Pipeline
Uses Groq API (GPT-OSS 120B) to annotate clauses from text files.

SETUP:
    pip install groq

USAGE:
    python annotate_clauses.py --input_dir data --output combined_annotations.jsonl --api_key YOUR_KEY

    Or set GROQ_API_KEY as environment variable and omit --api_key
"""

import os
import json
import time
import argparse
from groq import Groq

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal contract risk annotation assistant specialized in License Agreements.

The clause comes from a License Agreement, where:
- "Licensor" is the party that owns the intellectual property (IP) and grants usage rights
- "Licensee" is the party that receives the right to use the IP

----------------------------------------
CORE DECISION RULE
----------------------------------------
IF any HIGH trigger is present → High
ELSE IF any MEDIUM trigger is present → Medium
ELSE → Low

Always choose the highest applicable risk.
Risk = what the clause COSTS or RESTRICTS that party.
A clause that protects a party is Low for that party.
A clause that GRANTS a party a right or option is Low for that party —
risk only flows to the party SUBJECT TO the obligation or consequence.

----------------------------------------
HIGH RISK TRIGGERS
----------------------------------------
- Uncapped / unlimited liability or indemnification
- Unilateral termination without cause or compensation
- License revocable at will with no protection for Licensee
- IP ownership assigned to one party for work created by the other
- Usage restrictions that prevent Licensee's core business operations
- Unilateral amendment rights with no notice or protection
- Automatic renewal with no reasonable opt-out
- Immediate cessation of use with no wind-down, where Licensee's
  operations depend on the licensed IP

----------------------------------------
MEDIUM RISK TRIGGERS
----------------------------------------
- Liability with a defined cap (Medium for the party whose recovery
  is limited, Low for the party being protected)
- Early or unilateral termination with notice (NOT fixed-term expiry)
- Compliance obligation where failure is a "material breach"
  (exposes that party to license termination)
- Confidentiality obligations that extend beyond the contract term
- Governing law in a distant or unfamiliar jurisdiction
- Royalty or payment obligations with a defined rate or cap
- Audit rights (Medium for the party being audited only)
- Post-termination obligations (cease use, destroy materials)
- Change of control clause
- Wind-down period after termination (reasonable transition = Medium)
- Full warranty disclaimer (eliminates Licensee's right to recourse
  if licensed IP is defective — Medium for Licensee, Low for Licensor)
- Waiver of a default legal protection (e.g. rule that ambiguous
  terms are read against the drafter — removes Licensee's safety net)

----------------------------------------
INSTRUCTIONS
----------------------------------------
You MUST follow this two-step process for every clause:

STEP 1 — SCRATCHPAD (required, plain text)
Write your reasoning before producing any JSON. Answer these four
questions explicitly:
  1. What does this clause specifically cost or restrict the Licensor?
     If it only gives the Licensor a right or protection, say so.
  2. What does this clause specifically cost or restrict the Licensee?
     If it only gives the Licensee a right or protection, say so.
  3. Which triggers apply to each party?
  4. Are there any safeguards (notice requirements, equal treatment,
     wind-down periods, caps) that downgrade a trigger?

STEP 2 — JSON OUTPUT
Only after completing the scratchpad, produce the JSON output.
Your JSON labels must be consistent with your scratchpad reasoning.
If your scratchpad says a clause gives a party a right, that party
cannot be High or Medium in the JSON.

Additional rules:
- Evaluate each party INDEPENDENTLY. After finishing the Licensor,
  reset your reasoning before evaluating the Licensee. Do not let
  your conclusion for one party influence the other.
- Do NOT classify a clause as Low because it looks like a familiar
  clause type. Always evaluate the specific consequences in the text.
- Keep explanations to 1–2 plain English sentences. No legal jargon.

----------------------------------------
EXAMPLES
----------------------------------------

Example 1 — One-sided indemnification (Licensee bears risk):

Clause: "Licensee shall indemnify Licensor against any and all claims
arising from use of the licensed IP."

SCRATCHPAD:
1. Licensor: no cost — they are fully protected by the indemnity.
2. Licensee: must pay all damages and legal costs if any claim arises.
3. Licensee triggers: uncapped indemnification → High.
   Licensor triggers: none — clause protects them.
4. No safeguards present.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "The Licensor is fully covered — if any claim arises, the Licensee pays all costs."
    },
    "Licensee": {
      "risk": "High",
      "explanation": "The Licensee must cover all damages and legal costs if anything goes wrong, with no limit on the amount."
    }
  }
}

---

Example 2 — Right vs. burden (termination on insolvency):

Clause: "Either Party may immediately terminate this Agreement if the
other Party becomes insolvent or unable to pay its debts as they mature."

SCRATCHPAD:
1. Licensor: this clause gives the Licensor a right to exit if the
   Licensee becomes insolvent. Holding a right is not a cost or
   restriction — it is a protection.
2. Licensee: can lose the license immediately if it becomes insolvent,
   disrupting its operations.
3. Licensor triggers: none — the clause grants them a right, not an
   obligation.
   Licensee triggers: event-triggered termination → Medium.
4. No safeguards reduce the Licensee's exposure.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "This clause gives the Licensor the right to exit if the Licensee can't pay its bills — having that right is a protection, not a risk."
    },
    "Licensee": {
      "risk": "Medium",
      "explanation": "If the Licensee becomes insolvent, the Licensor can end the agreement immediately, meaning the Licensee could lose access to the IP at the worst possible time."
    }
  }
}

---

Example 3 — Mutual obligation evaluated independently:

Clause: "Each Party shall maintain in confidence all Confidential
Information of the other Party and shall not disclose it to any third
party. This obligation survives termination of this Agreement."

SCRATCHPAD:
1. Licensor: is also a Recipient — must keep the Licensee's
   confidential information secret even after the contract ends.
   This is a real ongoing obligation, not a protection.
2. Licensee: is also a Recipient — the exact same obligation applies.
   Evaluated independently of the Licensor.
3. Both parties: confidentiality beyond contract term → Medium for both.
4. No caps or time limit on survival — no downgrade applies.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Medium",
      "explanation": "The Licensor must keep the Licensee's confidential information secret indefinitely, even after the agreement ends — this is an ongoing obligation with no end date."
    },
    "Licensee": {
      "risk": "Medium",
      "explanation": "The Licensee must also keep the Licensor's confidential information secret indefinitely after the agreement ends — the same ongoing burden applies."
    }
  }
}

---

Example 4 — Safeguard downgrades a trigger:

Clause: "Licensee agrees to use the Licensed Domain Names only in
accordance with such content distribution policy that Licensor uses in
connection with its own business, and as may be established by Licensor
and communicated in writing in advance to Licensee, provided that
Licensee shall be afforded the same period of time to implement any
such policy as is afforded to Licensor's Affiliates and other third
parties."

SCRATCHPAD:
1. Licensor: can update its own policies normally — no new cost or
   restriction imposed on them.
2. Licensee: must follow Licensor's changing policies — surface risk
   of a unilateral change right.
3. Licensee triggers: unilateral policy change → initially looks High.
   Licensor triggers: none.
4. Safeguards present: written notice required in advance AND equal
   implementation time as Licensor's own affiliates → downgrade
   from High to Medium.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "The Licensor can update its policies as part of running its own business and faces no new cost or restriction."
    },
    "Licensee": {
      "risk": "Medium",
      "explanation": "The Licensor can change the rules the Licensee must follow, but must give written notice in advance and give the Licensee the same time to adapt as its own affiliates — so the risk is real but limited."
    }
  }
}

---

Example 5 — Material breach designation:

Clause: "Licensee shall remove any offending Content from the websites
as soon as possible after becoming aware of it. Licensee's failure to
comply with this Section shall be deemed a material breach of this
Agreement."

SCRATCHPAD:
1. Licensor: only gains a monitoring and enforcement right — no cost
   or restriction imposed on them.
2. Licensee: must comply with a content removal obligation. Failure
   is explicitly called a material breach, which can trigger
   termination of the entire agreement.
3. Licensee triggers: compliance obligation + material breach
   designation → Medium (termination exposure).
   Licensor triggers: none — clause gives them a right.
4. No safeguards reduce the Licensee's exposure.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "The Licensor gains the right to enforce content standards — this is a protection, not a cost."
    },
    "Licensee": {
      "risk": "Medium",
      "explanation": "The Licensee must remove problematic content quickly, and failing to do so counts as a serious breach that could result in losing the license."
    }
  }
}

----------------------------------------
OUTPUT FORMAT (STRICT — follow exactly)
----------------------------------------
First write your SCRATCHPAD as plain text.
Then produce the JSON block below with no extra text and no markdown
backticks around it.

{
  "clause_text": "<original clause>",
  "parties": {
    "Licensor": {
      "risk": "High | Medium | Low",
      "explanation": "<plain language explanation>"
    },
    "Licensee": {
      "risk": "High | Medium | Low",
      "explanation": "<plain language explanation>"
    }
  }
}

----------------------------------------
NOW ANALYZE:
----------------------------------------
{{clause_text}}"""


# ─────────────────────────────────────────────
# LOAD CLAUSES
# ─────────────────────────────────────────────

def load_clauses(filepath):
    """
    Reads a text file and splits into clauses.
    Each paragraph (separated by blank line) is treated as one clause.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on double newlines (paragraph breaks)
    raw_clauses = content.strip().split("\n\n")

    # Clean each clause
    clauses = []
    for clause in raw_clauses:
        cleaned = clause.strip().replace("\n", " ")
        if len(cleaned) > 20:  # skip very short fragments
            clauses.append(cleaned)

    return clauses


# ─────────────────────────────────────────────
# ANNOTATE SINGLE CLAUSE
# ─────────────────────────────────────────────

def annotate_clause(client, clause_text, clause_index):
    """
    Sends a single clause to Groq and returns parsed JSON annotation.
    """
    user_message = f"NOW ANALYZE:\n\n{clause_text}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # best model on Groq
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,             # low = consistent/deterministic
            max_tokens=2048,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        # Add metadata
        parsed["clause_index"] = clause_index
        parsed["status"] = "success"
        return parsed

    except json.JSONDecodeError as e:
        print(f"  [!] JSON parse error on clause {clause_index}: {e}")
        return {
            "clause_index": clause_index,
            "clause_text": clause_text,
            "status": "json_error",
            "raw_response": raw if 'raw' in locals() else ""
        }

    except Exception as e:
        print(f"  [!] API error on clause {clause_index}: {e}")
        return {
            "clause_index": clause_index,
            "clause_text": clause_text,
            "status": "api_error",
            "error": str(e)
        }


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(input_dir, output_file, api_key, delay=2.0):
    """
    Full annotation pipeline:
    1. Load clauses from all text files in the input directory
    2. Annotate each clause via Groq
    3. Save results to a single JSONL file (one JSON per line)
    """

    # Init client
    client = Groq(api_key=api_key)

    # Load clauses from all .txt files in the directory
    print(f"Scanning directory: {input_dir}")
    clauses = []
    
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            file_clauses = load_clauses(filepath)
            print(f"  Loaded {len(file_clauses)} clauses from {filename}")
            clauses.extend(file_clauses)
            
    print(f"\nFound {len(clauses)} total clauses across all files\n")

    if not clauses:
        print("No clauses found. Exiting.")
        return

    # Track stats
    success_count = 0
    error_count = 0

    # Open output file
    with open(output_file, "w", encoding="utf-8") as out_f:

        for i, clause in enumerate(clauses):
            print(f"Annotating clause {i+1}/{len(clauses)}...")

            result = annotate_clause(client, clause, clause_index=i+1)

            # Write to JSONL (one JSON object per line)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()  # write immediately, don't buffer

            if result["status"] == "success":
                success_count += 1
                # Print a quick preview
                parties = result.get("parties", {})
                for party, data in parties.items():
                    print(f"   {party}: {data['risk']}")
            else:
                error_count += 1
                print(f"   Status: {result['status']}")

            # Respect rate limits — 30 requests/min on free tier
            # 2 second delay = safe at ~30 req/min
            if i < len(clauses) - 1:
                time.sleep(delay)

    # Summary
    print(f"\n{'='*40}")
    print(f"Annotation complete.")
    print(f"  Successful : {success_count}")
    print(f"  Errors     : {error_count}")
    print(f"  Output     : {output_file}")
    print(f"{'='*40}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal clause risk annotator using Groq API")
    parser.add_argument("--input_dir", required=False, default="data", help="Path to input folder containing txt files (default: data)")
    parser.add_argument("--output",    required=True,  help="Path to output JSONL file")
    parser.add_argument("--api_key",   required=False, help="Groq API key (or set GROQ_API_KEY env variable)")
    parser.add_argument("--delay",     required=False, type=float, default=2.0, help="Delay between API calls in seconds (default: 2.0)")

    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Use --api_key or set GROQ_API_KEY environment variable.")

    run_pipeline(
        input_dir=args.input_dir,
        output_file=args.output,
        api_key=api_key,
        delay=args.delay
    )