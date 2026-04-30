import json
from collections import Counter

def check_text_duplicates(file_path):
    text_counts = Counter()
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            text_counts[text] += 1
            
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    print(f"Total entries: {sum(text_counts.values())}")
    print(f"Unique texts: {len(text_counts)}")
    print(f"Number of texts with duplicates: {len(duplicates)}")
    
    if duplicates:
        print("\nTop 5 most frequent duplicate texts (first 100 chars):")
        for text, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  [{count} times] {text[:100]}...")

check_text_duplicates('val.jsonl')
