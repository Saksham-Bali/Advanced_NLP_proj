import json
import collections

def count_unique_clauses(file_path):
    unique_clauses = set()
    total_samples = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                
                # Logic to extract clause body between triple quotes
                if 'Clause:\n"""' in text:
                    clause_body = text.split('Clause:\n"""')[1].split('"""')[0].strip()
                else:
                    # Fallback if the format is slightly different
                    clause_body = text
                
                unique_clauses.add(clause_body)
                total_samples += 1
            except Exception as e:
                print(f"Error parsing line: {e}")
                
    return len(unique_clauses), total_samples

if __name__ == "__main__":
    file_to_check = 'train.jsonl'
    unique_count, total_count = count_unique_clauses(file_to_check)
    
    print(f"File: {file_to_check}")
    print(f"Total samples: {total_count}")
    print(f"Unique clauses: {unique_count}")
    if unique_count > 0:
        print(f"Average samples per clause: {total_count / unique_count:.2f}")
