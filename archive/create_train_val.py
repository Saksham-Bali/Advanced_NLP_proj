import json
import random
import argparse
from collections import defaultdict

def create_train_val_split(input_file, train_output, val_output):
    print("Loading data and grouping by clause to maintain Licensor/Licensee pairs...")
    
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

    # ----------------------------------------------------
    # 1. CREATE VALIDATION SET FIRST (To prevent data leakage)
    # ----------------------------------------------------
    # We take a fixed chunk of clauses from each bucket for validation.
    # We will grab 25 High, 50 Medium, and 50 Low clauses (Total 125 clauses = 250 rows)
    val_high = 25
    val_med = 50
    val_low = 50
    
    val_clauses = []
    
    # Pop them out of the main pools so they can't be used in training
    for _ in range(val_high):
        if high_clauses: val_clauses.append(high_clauses.pop(random.randrange(len(high_clauses))))
        
    for _ in range(val_med):
        if medium_clauses: val_clauses.append(medium_clauses.pop(random.randrange(len(medium_clauses))))
        
    for _ in range(val_low):
        if low_clauses: val_clauses.append(low_clauses.pop(random.randrange(len(low_clauses))))

    val_dataset = []
    for rows in val_clauses:
        val_dataset.extend([r.copy() for r in rows])
        
    # ----------------------------------------------------
    # 2. CREATE TRAINING SET (Targeting 2500 rows / 1250 clauses)
    # ----------------------------------------------------
    target_train_clauses = 1250
    train_dataset = []
    
    copies_per_high = 3
    print(f"\n---> Sampling Validation Set: {len(val_clauses)} pairs held out completely.")
    print(f"---> Adding Training High risk clauses with {copies_per_high}x exact duplication...")
    
    high_count_added = 0
    for rows in high_clauses:
        for _ in range(copies_per_high):
            train_dataset.extend([r.copy() for r in rows])
            high_count_added += 1

    remaining_slots = target_train_clauses - high_count_added
    print(f"---> Filling remaining {remaining_slots} Training clause slots using Medium risk clauses...")
    
    if remaining_slots <= len(medium_clauses):
        sampled_mediums = random.sample(medium_clauses, remaining_slots)
        for rows in sampled_mediums:
            train_dataset.extend([r.copy() for r in rows])
    else:
        for rows in medium_clauses:
            train_dataset.extend([r.copy() for r in rows])
        spillover = remaining_slots - len(medium_clauses)
        sampled_lows = random.sample(low_clauses, spillover)
        for rows in sampled_lows:
            train_dataset.extend([r.copy() for r in rows])

    # Shuffle both datasets
    print("\nShuffling and saving...")
    random.shuffle(val_dataset)
    random.shuffle(train_dataset)

    # Save Train
    with open(train_output, 'w') as f:
        for item in train_dataset: f.write(json.dumps(item) + '\n')
            
    # Save Val
    with open(val_output, 'w') as f:
        for item in val_dataset: f.write(json.dumps(item) + '\n')

    # Print Stats
    print("\n==================================")
    print("--- 📚 TRAIN DATASET STATS ---")
    counts_train = defaultdict(int)
    for item in train_dataset: counts_train[item['_meta']['risk']] += 1
    for risk in ['High', 'Medium', 'Low']: print(f"[{risk.upper()}]: {counts_train[risk]} items")
    print(f"Total Rows Saved -> {train_output}: {len(train_dataset)}")
    
    print("\n--- 📝 VALIDATION DATASET STATS ---")
    counts_val = defaultdict(int)
    for item in val_dataset: counts_val[item['_meta']['risk']] += 1
    for risk in ['High', 'Medium', 'Low']: print(f"[{risk.upper()}]: {counts_val[risk]} items")
    print(f"Total Rows Saved -> {val_output}: {len(val_dataset)}")
    print("==================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="finetune.jsonl")
    parser.add_argument("--train-out", type=str, default="train.jsonl")
    parser.add_argument("--val-out", type=str, default="val.jsonl")
    args = parser.parse_args()
    
    create_train_val_split(args.input, args.train_out, args.val_out)
