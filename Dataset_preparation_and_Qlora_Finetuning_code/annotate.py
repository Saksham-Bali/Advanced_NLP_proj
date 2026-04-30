"""
Legal Clause Risk Annotation Pipeline
Uses Groq API (openai/gpt-oss-120b) to annotate clauses from a text file.

SETUP:
    pip install groq

USAGE:
    python annotate_clauses.py --input_file clauses.txt --output final.json --api_key YOUR_KEY

    Or set GROQ_API_KEY as environment variable and omit --api_key

RESUME:    Skips clauses already in final.json (matched by clause_index).
RETRY:     Rate-limit -> waits 15 s, retries once. Second rate-limit -> stops execution entirely.
SKIP:      Context-too-long -> prints full prompt to terminal, skips clause, no write.
"""

import os
import sys
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
IF any HIGH trigger is present -> High
ELSE -> Low

Always choose the highest applicable risk.

Risk = what the clause COSTS or RESTRICTS that party.
A clause that protects a party is Low for that party.
A clause that GRANTS a party a right or option is Low for that party --
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
IMPORTANT RULE FOR REMOVED MEDIUM CASES
----------------------------------------
Any clause that would previously be Medium should now be treated as:
-> Low, UNLESS it clearly creates severe, open-ended, or highly restrictive risk

----------------------------------------
ALWAYS LOW — DO NOT UPGRADE THESE
----------------------------------------
The following clause types are ALWAYS Low for both parties:
- Pure definitions ("X means ...", "shall have the meaning ...")
- Severability clauses
- Signature blocks and execution formalities
- Headings / interpretation rules
- Counterparts clauses
- Entire agreement / merger clauses
- Standard confidentiality exceptions
- No-agency / independent contractor clauses
- Notices

----------------------------------------
CONSISTENCY RULE FOR MIRROR CLAUSES
----------------------------------------
If a clause is mutual, both parties MUST receive the SAME risk level unless obligations differ materially.

----------------------------------------
INSTRUCTIONS
----------------------------------------
You MUST follow this two-step process for every clause:

STEP 1 -- SCRATCHPAD (required, plain text)
1. What does this clause cost or restrict the Licensor?
2. What does this clause cost or restrict the Licensee?
3. Does any HIGH trigger apply to either party?
4. Are there safeguards (caps, notice, limits) that reduce severity?

STEP 2 -- JSON OUTPUT

{
  "clause_text": "<original clause>",
  "parties": {
    "Licensor": {
      "risk": "High | Low",
      "explanation": "<plain language explanation>"
    },
    "Licensee": {
      "risk": "High | Low",
      "explanation": "<plain language explanation>"
    }
  }
}
"""


# ─────────────────────────────────────────────
# LOAD CLAUSES
# ─────────────────────────────────────────────

def load_clauses(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    raw_clauses = content.strip().split("\n\n")
    clauses = []
    for clause in raw_clauses:
        cleaned = clause.strip().replace("\n", " ")
        if len(cleaned) > 20:
            clauses.append(cleaned)
    return clauses


# ─────────────────────────────────────────────
# LOAD ALREADY-DONE INDICES
# ─────────────────────────────────────────────

def load_done_indices(output_file):
    done = set()
    if not os.path.exists(output_file):
        return done
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            if "clause_index" in entry:
                done.add(entry["clause_index"])
        print(f"Resuming -- {len(done)} clause(s) already annotated.")
    except (json.JSONDecodeError, ValueError):
        print(f"[!] Could not parse existing {output_file}. Starting fresh.")
    return done


# ─────────────────────────────────────────────
# APPEND RESULT TO output file
# ─────────────────────────────────────────────

def append_result(output_file, result):
    data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = []
    data.append(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# ERROR HELPERS
# ─────────────────────────────────────────────

def is_rate_limit_error(exc):
    msg = str(exc).lower()
    return (
        "rate_limit" in msg
        or "rate limit" in msg
        or "ratelimit" in msg
        or "429" in msg
        or "too many requests" in msg
    )

def is_context_length_error(exc):
    msg = str(exc).lower()
    return (
        "context_length_exceeded" in msg
        or "tokens_limit_reached" in msg
        or ("context" in msg and "length" in msg)
        or "maximum context" in msg
        or ("too long" in msg)
        or ("reduce" in msg and "token" in msg)
    )


# ─────────────────────────────────────────────
# EXTRACT JSON FROM MODEL RESPONSE
# ─────────────────────────────────────────────

def extract_json(raw_text):
    brace_count = 0
    end_idx = None
    start_idx = None

    for i in range(len(raw_text) - 1, -1, -1):
        ch = raw_text[i]
        if ch == '}':
            if brace_count == 0:
                end_idx = i
            brace_count += 1
        elif ch == '{':
            brace_count -= 1
            if brace_count == 0:
                start_idx = i
                break

    if start_idx is None or end_idx is None:
        raise ValueError("No JSON object found in model response")

    json_str = raw_text[start_idx : end_idx + 1]
    return json.loads(json_str)


# ─────────────────────────────────────────────
# ANNOTATE SINGLE CLAUSE
# ─────────────────────────────────────────────

def annotate_clause(client, clause_text, clause_index):
    """
    Returns a result dict, or None if the clause should be skipped entirely
    (context-too-long). Exits program on double rate-limit.
    """
    user_message = f"NOW ANALYZE:\n\n{clause_text}"
    raw = ""

    for attempt in range(1, 3):   # attempt 1, then attempt 2
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            parsed = extract_json(raw)
            parsed["clause_index"] = clause_index
            parsed["status"]       = "success"
            if "clause_text" not in parsed:
                parsed["clause_text"] = clause_text
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [!] Could not parse JSON for clause {clause_index}: {e}")
            print(f"  [!] Raw response:\n{raw}\n")
            return {
                "clause_index": clause_index,
                "clause_text":  clause_text,
                "status":       "json_error",
                "raw_response": raw,
            }

        except Exception as e:
            if is_rate_limit_error(e):
                if attempt == 1:
                    print(f"  [!] Rate limit on clause {clause_index}. Waiting 15 s then retrying...")
                    time.sleep(15)
                    continue
                else:
                    print(f"  [CRITICAL] Rate limit again on clause {clause_index}. Stopping execution entirely.")
                    sys.exit(1) # Stops doing anything

            if is_context_length_error(e):
                full_prompt = SYSTEM_PROMPT + "\n\nNOW ANALYZE:\n\n" + clause_text
                print(f"\n{'='*60}")
                print(f"[!] CONTEXT TOO LONG -- clause {clause_index} skipped (not written to output).")
                print(f"[FULL PROMPT BELOW]\n")
                print(full_prompt)
                print(f"\n{'='*60}\n")
                return None

            print(f"  [!] API error on clause {clause_index} (attempt {attempt}): {e}")
            return {
                "clause_index": clause_index,
                "clause_text":  clause_text,
                "status":       "api_error",
                "error":        str(e),
            }

    return None


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(input_file, output_file, api_key, delay=2.0):
    client = Groq(api_key=api_key)

    print(f"Loading clauses from : {input_file}")
    clauses = load_clauses(input_file)
    print(f"Total clauses found  : {len(clauses)}")

    if not clauses:
        print("No clauses found. Exiting.")
        return

    done_indices = load_done_indices(output_file)

    success_count = 0
    error_count   = 0
    skip_count    = 0

    for i, clause in enumerate(clauses):
        clause_index = i + 1

        if clause_index in done_indices:
            print(f"[{clause_index}/{len(clauses)}] Already done -- skipping.")
            continue

        print(f"[{clause_index}/{len(clauses)}] Annotating...")

        result = annotate_clause(client, clause, clause_index)

        if result is None:
            skip_count += 1
        else:
            append_result(output_file, result)
            if result["status"] == "success":
                success_count += 1
                for party, data in result.get("parties", {}).items():
                    print(f"   {party}: {data['risk']}")
            else:
                error_count += 1
                print(f"   Status: {result['status']}")

        if clause_index < len(clauses):
            time.sleep(delay)

    print(f"\n{'='*40}")
    print(f"Annotation complete.")
    print(f"  Successful : {success_count}")
    print(f"  Errors     : {error_count}")
    print(f"  Skipped    : {skip_count}")
    print(f"  Output     : {output_file}")
    print(f"{'='*40}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal clause risk annotator using Groq API")
    parser.add_argument("--input_file", required=False, default="clauses.txt",
                        help="Path to text file with all clauses (default: clauses.txt)")
    parser.add_argument("--output",     required=False, default="final.json",
                        help="Output JSON file (default: final.json)")
    parser.add_argument("--api_key",    required=False,
                        help="Groq API key (or set GROQ_API_KEY env variable)")
    parser.add_argument("--delay",      required=False, type=float, default=2.0,
                        help="Delay between API calls in seconds (default: 2.0)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Use --api_key or set GROQ_API_KEY env variable.")

    run_pipeline(
        input_file=args.input_file,
        output_file=args.output,
        api_key=api_key,
        delay=args.delay,
    )