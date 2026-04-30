import json

def inspect_clause(file_path, target_idx, target_party):
    matches = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            meta = data.get('_meta', {})
            if meta.get('clause_index') == target_idx and meta.get('party') == target_party:
                matches.append(data)
    
    print(f"Found {len(matches)} matches for Clause {target_idx}, Party {target_party}")
    for i, match in enumerate(matches):
        print(f"\n--- Match {i+1} ---")
        # Extract explanation from text
        text = match['text']
        if 'Explanation:' in text:
            explanation = text.split('Explanation:')[1].split('<|eot_id|>')[0].strip()
            print(f"Explanation: {explanation}")
        else:
            print("Explanation not found in text.")

inspect_clause('val.jsonl', 1, 'Licensee')
