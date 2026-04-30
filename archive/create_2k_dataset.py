import json
import random
import argparse
import os
import re
import time
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm

try:
    from groq import Groq
except ImportError:
    print("Please install groq: pip install groq")
    exit(1)

REPHRASE_PROMPT = """You are a legal editor. Rephrase the following legal explanation while preserving its EXACT meaning, risk assessment, and technical nuance.
- Keep the length similar (1-2 sentences).
- Do NOT change the core logic or the facts.
- Do NOT include any preamble, headers, or <scratchpad>. Just return the rephrased text.

Original Explanation: 
"{explanation}"
"""

def extract_explanation(text):
    match = re.search(r"Explanation:\s*(.*?)<\|eot_id\|>", text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

def inject_explanation(text, new_explanation):
    return re.sub(r"(Explanation:\s*)(.*?)(<\|eot_id\|>)$", rf"\g<1>{new_explanation}\g<3>", text, count=1, flags=re.DOTALL)

def get_rephrased_explanation(client, original_text, attempt=1):
    prompt = REPHRASE_PROMPT.format(explanation=original_text)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith('"') and content.endswith('"'): content = content[1:-1]
        return content
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            time.sleep(15 * attempt)
            if attempt < 5: return get_rephrased_explanation(client, original_text, attempt + 1)
        return None

def rephrase_row(client, row):
    base_expl = extract_explanation(row['text'])
    if not base_expl:
        return row.copy(), 0, 1
    rephrased = get_rephrased_explanation(client, base_expl)
    if not rephrased:
        return row.copy(), 0, 1
    
    new_row = row.copy()
    new_row['text'] = inject_explanation(row['text'], rephrased)
    return new_row, 1, 0

def process_clause(client, clause_rows):
    # Clause rows is a list of exactly 2 rows (Licensor, Licensee)
    # We want to rephrase both independenty
    new_clause_rows = []
    successes = 0
    failures = 0
    for row in clause_rows:
        new_r, s, f = rephrase_row(client, row)
        new_clause_rows.append(new_r)
        successes += s
        failures += f
    return new_clause_rows, successes, failures

def build_2k_dataset(input_file, output_file, groq_key):
    client = Groq(api_key=groq_key)

    # 1. Group rows by clause_index to preserve pairs
    clauses = defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            idx = data.get('_meta', {}).get('clause_index')
            if idx is not None:
                clauses[idx].append(data)

    # 2. Bucket clauses by their maximum risk
    high_clauses = []
    medium_clauses = []
    low_clauses = []

    for idx, rows in clauses.items():
        risks = [r.get('_meta', {}).get('risk') for r in rows]
        if 'High' in risks:
            high_clauses.append(rows)
        elif 'Medium' in risks:
            medium_clauses.append(rows)
        else:
            low_clauses.append(rows)

    print("\n--- Initial Clause Buckets ---")
    print(f"High risk clauses: {len(high_clauses)}")
    print(f"Medium risk clauses: {len(medium_clauses)}")
    print(f"Low risk clauses: {len(low_clauses)}")

    # 3. Strategy for 2k samples (1000 clauses):
    # - Take ALL High clauses (approx 272).
    # - Duplicate the High clauses ONCE (272 copies) and rephrase them.
    # - This takes up ~544 clause slots.
    # - Fill the remaining slots (1000 - 544 = 456) randomly from Medium clauses.
    # - This effectively keeps pairs intact, requires barely any API calls, and fixes the Low-class domination!
    
    total_target_clauses = 1000
    copies_per_high = 1  # 1 duplicate for every high clause
    
    final_dataset = []
    
    # Add original Highs
    for rows in high_clauses:
        final_dataset.extend(rows)
        
    print(f"\n---> Generating {len(high_clauses)} rephrased copies of the High Risk clauses with Groq...")
    
    total_successes = 0
    total_failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for rows in high_clauses:
            futures.append(executor.submit(process_clause, client, rows))
            
        with tqdm(total=len(futures), desc="Rephrasing High Pairs") as pbar:
            for future in concurrent.futures.as_completed(futures):
                new_augmented_rows, s, f = future.result()
                final_dataset.extend(new_augmented_rows)
                total_successes += s
                total_failures += f
                pbar.set_postfix(success=total_successes, err=total_failures)
                pbar.update(1)

    print(f"Finished rephrasing! Success: {total_successes} | Failures: {total_failures}")
    
    # Add remaining from Medium
    current_clause_count = len(high_clauses) * 2
    needed_clauses = total_target_clauses - current_clause_count
    
    print(f"\n---> Adding {needed_clauses} clauses from the Medium Bucket to fill the 2000 quota...")
    
    if needed_clauses <= len(medium_clauses):
        sampled_mediums = random.sample(medium_clauses, needed_clauses)
        for rows in sampled_mediums:
            final_dataset.extend(rows)
    else:
        # If we somehow need more than exist in medium, spillover to low
        for rows in medium_clauses:
            final_dataset.extend(rows)
        spillover = needed_clauses - len(medium_clauses)
        sampled_lows = random.sample(low_clauses, spillover)
        for rows in sampled_lows:
            final_dataset.extend(rows)
            
    print("\nShuffling and saving...")
    random.shuffle(final_dataset)

    with open(output_file, 'w') as f:
        for item in final_dataset:
            f.write(json.dumps(item) + '\n')

    print("\n==================================")
    print("--- Final Resulting Dataset ---")
    counts = defaultdict(int)
    for item in final_dataset:
        counts[item['_meta']['risk']] += 1
    for risk in ['High', 'Medium', 'Low']:
        print(f"[{risk.upper()}]: {counts[risk]} items")
    print(f"\nTotal Individual Rows: {len(final_dataset)}")
    print(f"File Saved to: {output_file}")
    print("==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="finetune.jsonl")
    parser.add_argument("--output", "-o", type=str, default="finetune_2k.jsonl")
    parser.add_argument("--groq-key", type=str, default=os.getenv("GROQ_API_KEY"))
    args = parser.parse_args()
    
    if not args.groq_key:
        print("Please set the GROQ_API_KEY environment variable or pass --groq-key YOUR_KEY")
        exit(1)
        
    build_2k_dataset(args.input, args.output, args.groq_key)
