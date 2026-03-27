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

Use these exact party roles in your output.

----------------------------------------
CORE DECISION RULE
----------------------------------------
IF any HIGH trigger is present → High  
ELSE IF any MEDIUM trigger is present → Medium  
ELSE → Low  

Always choose the highest applicable risk.

Risk is always evaluated from the perspective of what the clause COSTS or 
RESTRICTS that party.

A clause that protects a party is Low risk for that party, even if it contains 
a trigger that harms the other party.

If a clause creates no cost or restriction for a party, always label that 
party Low regardless of what triggers affect the other party.

----------------------------------------
HIGH RISK TRIGGERS
----------------------------------------
- Uncapped / unlimited liability  
- One-sided indemnification  
- Unilateral termination without cause  
- IP ownership assigned entirely to one party (especially if created by the other party)
- License revocable at will without protection or compensation for the Licensee  
- Usage restrictions that prevent the Licensee from conducting their core business
  or that apply indefinitely beyond the contract term without compensation
- Automatic renewal with no reasonable opt-out  
- Unilateral amendment rights  

----------------------------------------
MEDIUM RISK TRIGGERS
----------------------------------------
- Liability with a defined cap  
- Termination with notice (early/unilateral termination only — NOT fixed-term expiry)
- Confidentiality beyond contract term  
- Governing law in a distant jurisdiction  
- Royalty / payment obligations with a defined rate or cap  
- Audit rights  
- Post-termination obligations (e.g., destroy materials, cease use)
- Change of control clause  

----------------------------------------
TRIGGER CLARIFICATIONS
----------------------------------------
- Uncapped liability requires explicit liability with no financial limit.
  A liability cap is Medium risk only for the party whose recovery is limited,
  not for the party being protected.

- Termination with notice applies only to early or unilateral termination.
  A fixed-term license that naturally expires is Low risk.

- If the Licensee builds or improves IP but ownership goes entirely to the Licensor,
  this is HIGH risk for the Licensee.

- Royalty obligations with a defined rate or cap = Medium.
  Royalty obligations that are open-ended, escalating, or have no defined limit
  = High under uncapped liability.

- Audit rights are Medium risk only for the party whose books are being audited,
  not for the party conducting the audit.

- Immediate cessation of use with no wind-down period after termination is High
  risk for Licensee if their core operations depend on the licensed IP.
  A reasonable wind-down period = Medium risk.

- Unilateral policy change rights are High only when there is NO notice 
  requirement and NO equal-treatment protection. If the clause requires 
  written notice in advance AND gives the Licensee the same implementation 
  time as the Licensor's own affiliates, classify as Medium, not High.

----------------------------------------
LICENSE-SPECIFIC INTERPRETATION
----------------------------------------
When analyzing risk, consider:

- Licensor controls the IP and can restrict usage
- Licensee depends on the license to operate, sell, or build products

Risk increases when:

For Licensee:
- License can be revoked easily or without compensation
- Usage rights are unclear, narrow, or easily restricted
- Strong payment or royalty obligations exist
- Improvements or derivative works are owned fully by Licensor

For Licensor:
- Liability or indemnity obligations are imposed
- IP control is weakened or transferred
- Revenue is capped or restricted unfairly

IMPORTANT:
- Ownership staying with Licensor is NORMAL and NOT a risk by default
- Only classify as risk if there is imbalance, restriction, or cost based on triggers
- Do NOT assume risk unless a defined trigger is present

----------------------------------------
INSTRUCTIONS
----------------------------------------
- Evaluate BOTH parties: "Licensor" and "Licensee"
- Do NOT invent new triggers
- Be consistent and deterministic
- Write explanations in plain, simple English that a non-lawyer can understand
- Avoid legal jargon
- Focus on real-world consequences
- Keep explanations to 1–2 sentences maximum

----------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY — no extra text, no markdown backticks)
----------------------------------------
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
EXAMPLES
----------------------------------------
Example 1 — Licensee bears risk:

Clause:
"Licensee shall indemnify Licensor against any and all claims arising from use of the licensed IP."

Output:
{
  "clause_text": "Licensee shall indemnify Licensor against any and all claims arising from use of the licensed IP.",
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "The Licensor is fully protected — if anyone makes a claim, the Licensee has to cover all the costs."
    },
    "Licensee": {
      "risk": "High",
      "explanation": "If anything goes wrong when using the IP, the Licensee has to pay for all damages and legal costs, which could be very expensive."
    }
  }
}

Example 2 — Licensor bears risk:

Clause:
"Licensor shall indemnify Licensee against any third-party claims alleging that the licensed IP infringes any patent or copyright."

Output:
{
  "clause_text": "Licensor shall indemnify Licensee against any third-party claims alleging that the licensed IP infringes any patent or copyright.",
  "parties": {
    "Licensor": {
      "risk": "High",
      "explanation": "If someone claims the IP belongs to them or infringes their rights, the Licensor has to pay all legal costs — even if the case is expensive or drawn out."
    },
    "Licensee": {
      "risk": "Low",
      "explanation": "The Licensee is protected here — if there is an IP ownership dispute, the Licensor covers all costs, not the Licensee."
    }
  }
}

Example 3 — Unilateral change right with protective safeguard:

Clause:
"Licensee agrees to use the Licensed Domain Names only in accordance with 
such content distribution policy that Licensor uses in connection with its 
own business, and as may be established by Licensor and communicated in 
writing in advance to Licensee from time to time, provided that Licensee 
shall be afforded the same period of time to implement any such policy as 
is afforded to Licensor's Affiliates and other third parties."

Output:
{
  "clause_text": "...",
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "The Licensor can update its own policies normally and faces no new cost or restriction."
    },
    "Licensee": {
      "risk": "Medium",
      "explanation": "The Licensor can change the rules the Licensee must follow, but must give written notice in advance and give the Licensee the same time to adapt as its own affiliates get — so the risk is real but limited."
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