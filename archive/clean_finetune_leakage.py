import json
import re

def replace_case_insensitive(match):
    word = match.group(0)
    lower_word = word.lower()
    
    if lower_word == 'high':
        replacement = 'significant'
    elif lower_word == 'medium':
        replacement = 'moderate'
    elif lower_word == 'low':
        replacement = 'minimal'
    else:
        return word
        
    # Preserve original capitalization
    if word.isupper():
        return replacement.upper()
    elif word.istitle():
        return replacement.title()
    else:
        return replacement

def clean_leakage(input_file, output_file):
    print(f"Reading from {input_file}...")
    
    # Regex to match whole words only
    pattern = re.compile(r'\b(high|medium|low)\b', re.IGNORECASE)
    
    processed = 0
    modified = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line)
            text = data.get("text", "")
            
            # Split to ensure we ONLY modify the explanation, not the "Risk Level: " label
            if "Explanation: " in text:
                parts = text.split("Explanation: ", 1)
                prefix = parts[0] + "Explanation: "
                explanation_text = parts[1]
                
                # Apply replacement ONLY to the explanation part
                new_explanation, count = pattern.subn(replace_case_insensitive, explanation_text)
                
                if count > 0:
                    modified += 1
                    data["text"] = prefix + new_explanation
            
            outfile.write(json.dumps(data) + "\n")
            processed += 1
            
    print("-" * 40)
    print(f"Total examples processed : {processed}")
    print(f"Examples cleaned         : {modified}")
    print(f"Saved clean dataset to   : {output_file}")
    print("-" * 40)

if __name__ == "__main__":
    clean_leakage("finetune_merged.jsonl", "finetune_merged_clean.jsonl")
