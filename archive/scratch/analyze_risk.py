import json
from collections import Counter

def analyze_risk_distribution(file_path):
    risk_counts = Counter()
    oversampled_risks = Counter()
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            meta = data.get('_meta', {})
            risk = meta.get('risk')
            idx = meta.get('clause_index')
            risk_counts[risk] += 1

    # Find which indices are oversampled
    indices = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            indices.append(data['_meta']['clause_index'])
    
    index_counts = Counter(indices)
    oversampled_indices = {idx for idx, count in index_counts.items() if count > 2}

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            meta = data.get('_meta', {})
            if meta['clause_index'] in oversampled_indices:
                oversampled_risks[meta['risk']] += 1

    print(f"Overall risk distribution: {dict(risk_counts)}")
    print(f"Risk distribution in oversampled clauses: {dict(oversampled_risks)}")

analyze_risk_distribution('val.jsonl')
