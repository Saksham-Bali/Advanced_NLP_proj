import json
import random
import argparse
from collections import defaultdict

def create_2500_dataset(input_file, output_file):
    print("Loading data and grouping by clause to maintain Licensor/Licensee pairs...")
    
    # Read rows sequentially in pairs (Licensor, then Licensee)
    # Since convert.py always writes them in pairs, this bypasses any clause_index collisions.
    clauses = []
    current_pair = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            current_pair.append(data)
            
            if len(current_pair) == 2:
                clauses.append(current_pair)
                current_pair = []

    high_clauses = []
    medium_clauses = []
    low_clauses = []

    for rows in clauses:
        risks = [r.get('_meta', {}).get('risk') for r in rows]
        if 'High' in risks:
            high_clauses.append(rows)
        elif 'Medium' in risks:
            medium_clauses.append(rows)
        else:
            low_clauses.append(rows)

    print("\n--- Available Clause Buckets ---")
    print(f"High risk clauses: {len(high_clauses)}")
    print(f"Medium risk clauses: {len(medium_clauses)}")
    print(f"Low risk clauses: {len(low_clauses)}")

    # We want 1250 clauses exactly (which equals 2500 rows)
    target_clauses = 1250
    final_dataset = []

    # 1. Heavily overweight High clauses since they are the rarest.
    # Duplicating High clauses 3 times (yielding 3 total copies per clause)
    # 272 * 3 = 816 clauses.
    copies_per_high = 3
    print(f"\n---> Adding High risk clauses with {copies_per_high}x exact duplication...")
    high_count_added = 0
    for rows in high_clauses:
        for _ in range(copies_per_high):
            # Duplicate the rows exactly
            final_dataset.extend([r.copy() for r in rows])
            high_count_added += 1

    # 2. Fill the remaining spots with Medium clauses
    remaining_slots = target_clauses - high_count_added
    print(f"---> Filling remaining {remaining_slots} clause slots using Medium risk clauses...")
    
    if remaining_slots <= len(medium_clauses):
        sampled_mediums = random.sample(medium_clauses, remaining_slots)
        for rows in sampled_mediums:
            final_dataset.extend([r.copy() for r in rows])
    else:
        # If we need more than we have, take all medium and spill over to low
        for rows in medium_clauses:
            final_dataset.extend([r.copy() for r in rows])
        spillover = remaining_slots - len(medium_clauses)
        print(f"---> Falling back: taking {spillover} slots from Low risk clauses to fill quota...")
        sampled_lows = random.sample(low_clauses, spillover)
        for rows in sampled_lows:
            final_dataset.extend([r.copy() for r in rows])

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
    print(f"\nTotal Individual Rows: {len(final_dataset)} (Extracted from 1250 paired clauses)")
    print(f"File Saved to: {output_file}")
    print("==================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="finetune.jsonl")
    parser.add_argument("--output", "-o", type=str, default="finetune_2500_exact.jsonl")
    args = parser.parse_args()
    
    create_2500_dataset(args.input, args.output)
