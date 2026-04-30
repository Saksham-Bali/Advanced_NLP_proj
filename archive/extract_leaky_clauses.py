import json
import re

def extract_leaky_clauses(input_file, leaky_output, clean_output):
    """
    Reads the annotation JSON, separates clauses that have the words 
    low, medium, or high in the reasoning/explanation for ANY party,
    and saves them into separate files.
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    leaky_clauses = []
    clean_clauses = []
    
    # Regex to match whole words: low, medium, high (case-insensitive)
    pattern = re.compile(r'\b(low|medium|high)\b', re.IGNORECASE)

    for record in data:
        is_leaky = False
        
        # Check the explanation for every party in this clause
        if "parties" in record:
            for party, info in record["parties"].items():
                explanation = info.get("explanation", "")
                if pattern.search(explanation):
                    is_leaky = True
                    break # If one party leaks it, the whole clause is considered leaky
        
        if is_leaky:
            leaky_clauses.append(record)
        else:
            clean_clauses.append(record)

    # Save the separated datasets
    with open(leaky_output, 'w') as f:
        json.dump(leaky_clauses, f, indent=2)
        
    with open(clean_output, 'w') as f:
        json.dump(clean_clauses, f, indent=2)

    print("-" * 30)
    print(f"Total clauses processed : {len(data)}")
    print(f"Leaky clauses extracted : {len(leaky_clauses)} -> Saved to {leaky_output}")
    print(f"Clean clauses remaining : {len(clean_clauses)} -> Saved to {clean_output}")
    print("-" * 30)

if __name__ == "__main__":
    input_filename = "completed_annotation.json"
    leaky_filename = "leaky_annotations.json"
    clean_filename = "clean_annotations.json"
    
    extract_leaky_clauses(input_filename, leaky_filename, clean_filename)
