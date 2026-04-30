import json
from collections import Counter

def analyze_jsonl(file_path):
    counts = Counter()
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            meta = data.get('_meta', {})
            party = meta.get('party')
            risk = meta.get('risk')
            counts[(party, risk)] += 1
    
    print(f"Distribution in {file_path}:")
    for (party, risk), count in sorted(counts.items()):
        print(f"  {party} - {risk}: {count}")

analyze_jsonl('finetune_final.jsonl')
