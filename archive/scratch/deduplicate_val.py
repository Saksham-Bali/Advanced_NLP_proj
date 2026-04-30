import json

def deduplicate_val(file_path):
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    unique_entries = {} # (clause_content, party) -> entry
    
    for entry in entries:
        text = entry.get('text', '')
        if 'Clause:\n"""' in text:
            clause = text.split('Clause:\n"""')[1].split('"""\n\nProvide:')[0].strip()
        else:
            # Fallback if text format is different, use clause_index as proxy but content is better
            clause = entry['_meta'].get('clause_index', 'unknown')
            
        party = entry['_meta'].get('party')
        key = (clause, party)
        
        if key not in unique_entries:
            unique_entries[key] = entry
            
    print(f"Original entries in {file_path}: {len(entries)}")
    print(f"Unique entries kept: {len(unique_entries)}")
    
    with open(file_path, 'w') as f:
        for entry in unique_entries.values():
            f.write(json.dumps(entry) + '\n')

deduplicate_val('val.jsonl')
