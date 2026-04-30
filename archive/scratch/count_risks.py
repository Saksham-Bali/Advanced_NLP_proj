import json
from collections import Counter

counts = Counter()
with open('finetune.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            meta = data.get('_meta', {})
            party = meta.get('party')
            risk = meta.get('risk')
            counts[(party, risk)] += 1

print(f"{'Party':<12} | {'Risk':<10} | {'Count':<6}")
print("-" * 35)
for (party, risk), count in sorted(counts.items()):
    print(f"{party:<12} | {risk:<10} | {count:<6}")
