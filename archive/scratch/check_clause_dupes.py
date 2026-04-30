import json
from collections import Counter

def check_clause_content_duplicates(file_path):
    clause_counts = Counter()
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            if 'Clause:\n"""' in text:
                clause = text.split('Clause:\n"""')[1].split('"""\n\nProvide:')[0].strip()
                clause_counts[clause] += 1
            
    duplicates = {clause: count for clause, count in clause_counts.items() if count > 2} # Count > 2 because 2 is expected (Licensor + Licensee)
    print(f"Total entries: {sum(clause_counts.values())}")
    print(f"Unique clause contents: {len(clause_counts)}")
    print(f"Number of clauses appearing more than twice: {len(duplicates)}")
    
    if duplicates:
        print("\nOversampled clauses (appearing > 2 times):")
        for clause, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  [{count} times] {clause[:100]}...")

check_clause_content_duplicates('val.jsonl')
