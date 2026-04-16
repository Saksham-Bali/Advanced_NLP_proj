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
ELSE IF any MEDIUM trigger is present -> Medium
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
  if licensed IP is defective -- Medium for Licensee, Low for Licensor)
- Waiver of a default legal protection (e.g. rule that ambiguous
  terms are read against the drafter -- removes Licensee's safety net)

----------------------------------------
INSTRUCTIONS
----------------------------------------
You MUST follow this two-step process for every clause:

STEP 1 -- SCRATCHPAD (required, plain text)
Write your reasoning before producing any JSON. Answer these four
questions explicitly:
  1. What does this clause specifically cost or restrict the Licensor?
     If it only gives the Licensor a right or protection, say so.
  2. What does this clause specifically cost or restrict the Licensee?
     If it only gives the Licensee a right or protection, say so.
  3. Which triggers apply to each party?
  4. Are there any safeguards (notice requirements, equal treatment,
     wind-down periods, caps) that downgrade a trigger?

STEP 2 -- JSON OUTPUT
Only after completing the scratchpad, produce the JSON output.
Your JSON labels must be consistent with your scratchpad reasoning.
If your scratchpad says a clause gives a party a right, that party
cannot be High or Medium in the JSON.

Additional rules:
- Evaluate each party INDEPENDENTLY.
- Do NOT classify a clause as Low because it looks like a familiar clause type.
- Keep explanations to 1-2 plain English sentences. No legal jargon.

----------------------------------------
EXAMPLES
----------------------------------------

Example 1:
Clause: "Licensee shall indemnify Licensor against any and all claims arising from use of the licensed IP."

SCRATCHPAD:
1. Licensor: no cost -- fully protected by the indemnity.
2. Licensee: must pay all damages and legal costs if any claim arises.
3. Licensee triggers: uncapped indemnification -> High. Licensor triggers: none.
4. No safeguards present.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "The Licensor is fully covered -- if any claim arises, the Licensee pays all costs."
    },
    "Licensee": {
      "risk": "High",
      "explanation": "The Licensee must cover all damages and legal costs if anything goes wrong, with no limit on the amount."
    }
  }
}

---

Example 2:
Clause: "Either Party may immediately terminate this Agreement if the other Party becomes insolvent."

SCRATCHPAD:
1. Licensor: gains a right to exit -- a protection, not a cost.
2. Licensee: can lose the license immediately if it becomes insolvent.
3. Licensor triggers: none. Licensee triggers: event-triggered termination -> Medium.
4. No safeguards reduce the Licensee's exposure.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Low",
      "explanation": "This clause gives the Licensor the right to exit if the Licensee cannot pay its bills -- a protection, not a risk."
    },
    "Licensee": {
      "risk": "Medium",
      "explanation": "If the Licensee becomes insolvent, the Licensor can end the agreement immediately."
    }
  }
}

---

Example 3:
Clause: "Each Party shall maintain in confidence all Confidential Information of the other Party. This obligation survives termination."

SCRATCHPAD:
1. Licensor: must keep Licensee's confidential info secret even after contract ends -- a real ongoing obligation.
2. Licensee: same obligation applies independently.
3. Both parties: confidentiality beyond contract term -> Medium for both.
4. No time limit on survival -- no downgrade.

OUTPUT:
{
  "parties": {
    "Licensor": {
      "risk": "Medium",
      "explanation": "The Licensor must keep the Licensee's confidential information secret indefinitely after the agreement ends."
    },
    "Licensee": {
      "risk": "Medium",
      "explanation": "The Licensee must also keep the Licensor's confidential information secret indefinitely after the agreement ends."
    }
  }
}

----------------------------------------
OUTPUT FORMAT (STRICT)
----------------------------------------
First write your SCRATCHPAD as plain text.
Then output the JSON object. Do NOT wrap the JSON in markdown backticks.

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
}"""


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