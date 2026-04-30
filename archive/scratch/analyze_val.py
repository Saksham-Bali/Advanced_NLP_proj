import json
from collections import Counter

def analyze_val_jsonl(file_path):
    indices = []
    party_pairs = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            meta = data.get('_meta', {})
            idx = meta.get('clause_index')
            party = meta.get('party')
            risk = meta.get('risk')
            
            if idx is not None:
                indices.append(idx)
                party_pairs.append((idx, party))

    index_counts = Counter(indices)
    party_pair_counts = Counter(party_pairs)
    
    print(f"Total entries: {len(indices)}")
    print(f"Unique clause indices: {len(index_counts)}")
    
    # Distribution of counts per index
    counts_freq = Counter(index_counts.values())
    print("\nDistribution of occurrences per clause_index:")
    for count, freq in sorted(counts_freq.items()):
        print(f"  {count} occurrences: {freq} clauses")

    # Check for duplicates (same index and party)
    duplicates = {pair: count for pair, count in party_pair_counts.items() if count > 1}
    if duplicates:
        print("\nDuplicates found (same clause_index and party):")
        for pair, count in sorted(duplicates.items())[:10]:
            print(f"  Clause {pair[0]}, Party {pair[1]}: {count} times")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more.")
    else:
        print("\nNo duplicates (same clause_index and party) found.")

    # Show some oversampled examples if they exist
    most_common = index_counts.most_common(10)
    print("\nTop 10 most frequent clause indices:")
    for idx, count in most_common:
        print(f"  Clause {idx}: {count} times")

analyze_val_jsonl('val.jsonl')
