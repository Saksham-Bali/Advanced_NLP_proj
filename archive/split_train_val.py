import json
import random
import argparse
from collections import defaultdict

def create_train_val_split(input_file, train_output, val_output, val_ratio=0.1):
    print("Loading data and grouping by clause_index to maintain Licensor/Licensee pairs...")
    
    # Group by clause_index
    clauses = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            clause_idx = data.get('_meta', {}).get('clause_index')
            if clause_idx is not None:
                clauses[clause_idx].append(data)
            else:
                print("Warning: Missing clause_index in data. Adding randomly.")
                clauses[random.randint(100000, 999999)].append(data)
                
    clause_groups = list(clauses.values())
    random.shuffle(clause_groups)
    
    num_val = int(len(clause_groups) * val_ratio)
    val_clauses = clause_groups[:num_val]
    train_clauses = clause_groups[num_val:]
    
    train_dataset = []
    for rows in train_clauses:
        train_dataset.extend(rows)
        
    val_dataset = []
    for rows in val_clauses:
        val_dataset.extend(rows)
        
    print("\nShuffling and saving...")
    random.shuffle(val_dataset)
    random.shuffle(train_dataset)

    # Save Train
    with open(train_output, 'w') as f:
        for item in train_dataset: 
            f.write(json.dumps(item) + '\n')
            
    # Save Val
    with open(val_output, 'w') as f:
        for item in val_dataset: 
            f.write(json.dumps(item) + '\n')

    # Print Stats
    print("\n==================================")
    print("--- 📚 TRAIN DATASET STATS ---")
    counts_train = defaultdict(int)
    for item in train_dataset: 
        counts_train[item['_meta']['risk']] += 1
    for risk in sorted(counts_train.keys()): 
        print(f"[{risk.upper()}]: {counts_train[risk]} items")
    print(f"Total Rows Saved -> {train_output}: {len(train_dataset)}")
    
    print("\n--- 📝 VALIDATION DATASET STATS ---")
    counts_val = defaultdict(int)
    for item in val_dataset: 
        counts_val[item['_meta']['risk']] += 1
    for risk in sorted(counts_val.keys()): 
        print(f"[{risk.upper()}]: {counts_val[risk]} items")
    print(f"Total Rows Saved -> {val_output}: {len(val_dataset)}")
    print("==================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="finetune_final.jsonl")
    parser.add_argument("--train-out", type=str, default="train.jsonl")
    parser.add_argument("--val-out", type=str, default="val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    create_train_val_split(args.input, args.train_out, args.val_out, args.val_ratio)
