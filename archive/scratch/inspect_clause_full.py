import json

def inspect_clause_full(file_path, target_idx, target_party):
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
        text = match['text']
        # Extract clause content
        if 'Clause:' in text:
            clause = text.split('Clause:')[1].split('Provide:')[0].strip().replace('"""', '')
            print(f"Clause content: {clause[:150]}...")
        else:
            print("Clause content not found.")
        
        if 'Explanation:' in text:
            explanation = text.split('Explanation:')[1].split('<|eot_id|>')[0].strip()
            print(f"Explanation: {explanation}")

inspect_clause_full('val.jsonl', 1, 'Licensee')
