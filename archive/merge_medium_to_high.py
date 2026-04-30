import json
import argparse

def merge_labels(input_file, output_file):
    print(f"Reading from {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return
        
    converted_count_licensor = 0
    converted_count_licensee = 0
    
    for clause in data:
        if "parties" in clause:
            # Check Licensor
            if "Licensor" in clause["parties"]:
                if clause["parties"]["Licensor"].get("risk") == "Medium":
                    clause["parties"]["Licensor"]["risk"] = "High"
                    converted_count_licensor += 1
            
            # Check Licensee
            if "Licensee" in clause["parties"]:
                if clause["parties"]["Licensee"].get("risk") == "Medium":
                    clause["parties"]["Licensee"]["risk"] = "High"
                    converted_count_licensee += 1
                        
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print("-" * 40)
    print(f"Licensor 'Medium' -> 'High' conversions: {converted_count_licensor}")
    print(f"Licensee 'Medium' -> 'High' conversions: {converted_count_licensee}")
    print(f"Total conversions                        : {converted_count_licensor + converted_count_licensee}")
    print(f"Saved merged dataset to                  : {output_file}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Medium risk labels to High.")
    parser.add_argument("--input", default="completed_annotation.json", help="Input JSON file")
    parser.add_argument("--output", default="completed_annotation_merged.json", help="Output JSON file")
    args = parser.parse_args()
    
    merge_labels(args.input, args.output)
