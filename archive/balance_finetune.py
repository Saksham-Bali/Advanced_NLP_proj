import json
import random
import argparse
from collections import defaultdict

def balance_dataset(input_file, output_file, strategy="undersample"):
    """
    Balances the finetuning JSONL dataset so that Low, Medium, and High 
    risk examples have equal representation.
    """
    
    # 1. Load data and group by risk category
    categories = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Extract the risk label from metadata
            risk = data.get('_meta', {}).get('risk')
            if risk:
                categories[risk].append(data)

    print("--- Original Distribution ---")
    for risk, items in categories.items():
        print(f"{risk}: {len(items)}")
        
    if not categories:
        print("No valid data found.")
        return

    # 2. Determine target size based on strategy
    if strategy == "undersample":
        # Target size is the smallest category to avoid duplicates
        target_size = min(len(items) for items in categories.values())
        print(f"\nStrategy: Undersampling to smallest class size ({target_size})")
    elif strategy == "oversample":
        # Target size is the largest category, duplicating smaller classes
        target_size = max(len(items) for items in categories.values())
        print(f"\nStrategy: Oversampling to largest class size ({target_size})")
    else:
        raise ValueError("Invalid strategy. Use 'undersample' or 'oversample'.")

    # 3. Create balanced dataset
    balanced_data = []
    
    for risk, items in categories.items():
        if len(items) == target_size:
            sampled = items
        elif len(items) > target_size:
            # Undersample: pick random subset
            sampled = random.sample(items, target_size)
        else:
            # Oversample: randomly duplicate items until target size is reached
            sampled = items.copy()
            while len(sampled) < target_size:
                sampled.append(random.choice(items))
                
        balanced_data.extend(sampled)

    # 4. Shuffle the final combined dataset so classes are mixed during training
    random.shuffle(balanced_data)

    # 5. Write to output
    with open(output_file, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item) + '\n')

    print("\n--- Balanced Distribution (Saved) ---")
    counts = defaultdict(int)
    for item in balanced_data:
        counts[item['_meta']['risk']] += 1
    
    for risk, count in counts.items():
        print(f"{risk}: {count}")
    
    print(f"\nTotal examples: {len(balanced_data)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance the finetune JSONL dataset.")
    parser.add_argument("--input", "-i", type=str, default="finetune.jsonl", help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, default="finetune_balanced.jsonl", help="Output JSONL file")
    parser.add_argument("--strategy", "-s", type=str, choices=["undersample", "oversample"], 
                        default="undersample", help="Balancing strategy")
    
    args = parser.parse_args()
    
    balance_dataset(args.input, args.output, args.strategy)
