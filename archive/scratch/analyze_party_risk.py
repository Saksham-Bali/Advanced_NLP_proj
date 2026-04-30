import json
from collections import Counter

def analyze_risk_per_party(file_path):
    # Dictionary to store counts: party -> risk -> count
    counts = {
        'Licensor': Counter(),
        'Licensee': Counter()
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                meta = data.get('_meta', {})
                party = meta.get('party')
                risk = meta.get('risk')
                
                if party in counts:
                    counts[party][risk] += 1
                else:
                    # Handle cases where party might be named differently or missing
                    if party:
                        if party not in counts:
                            counts[party] = Counter()
                        counts[party][risk] += 1
            except json.JSONDecodeError:
                continue

    print("Risk Distribution per Party in val.jsonl:")
    for party, risks in counts.items():
        print(f"\nParty: {party}")
        total = sum(risks.values())
        for risk, count in sorted(risks.items()):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {risk}: {count} ({percentage:.1f}%)")
        print(f"  Total: {total}")

analyze_risk_per_party('val.jsonl')
