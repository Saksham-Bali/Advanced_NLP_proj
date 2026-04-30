import json

f1 = 'final_test1.json'
f2 = 'final_test_updated.json'

with open(f1, 'r') as f: d1 = json.load(f)
with open(f2, 'r') as f: d2 = json.load(f)

d1_dict = {item.get('clause_index'): item for item in d1 if 'clause_index' in item and item.get("status") == "success"}
d2_dict = {item.get('clause_index'): item for item in d2 if 'clause_index' in item and item.get("status") == "success"}

common_idx = sorted(list(set(d1_dict.keys()).intersection(set(d2_dict.keys()))))
print("Examining key improvements:")
for idx in common_idx:
    c1 = d1_dict[idx]
    c2 = d2_dict[idx]
    r1_licensor = c1.get("parties", {}).get("Licensor", {}).get("risk", "Unknown")
    r2_licensor = c2.get("parties", {}).get("Licensor", {}).get("risk", "Unknown")
    r1_licensee = c1.get("parties", {}).get("Licensee", {}).get("risk", "Unknown")
    r2_licensee = c2.get("parties", {}).get("Licensee", {}).get("risk", "Unknown")
    
    # Check for Medium -> Low for Licensee
    if r1_licensee == "Medium" and r2_licensee == "Low":
        print(f"\n[Index {idx}] Medium -> Low")
        print("Text:", c1["clause_text"])
        print("F1 (Old) Exp:", c1.get("parties", {}).get("Licensee", {}).get("explanation"))
        print("F2 (New) Exp:", c2.get("parties", {}).get("Licensee", {}).get("explanation"))
        break

for idx in common_idx:
    c1 = d1_dict[idx]
    c2 = d2_dict[idx]
    r1_licensee = c1.get("parties", {}).get("Licensee", {}).get("risk", "Unknown")
    r2_licensee = c2.get("parties", {}).get("Licensee", {}).get("risk", "Unknown")
    if r1_licensee == "High" and r2_licensee == "Medium":
        print(f"\n[Index {idx}] High -> Medium")
        print("Text:", c1["clause_text"])
        print("F1 (Old) Exp:", c1.get("parties", {}).get("Licensee", {}).get("explanation"))
        print("F2 (New) Exp:", c2.get("parties", {}).get("Licensee", {}).get("explanation"))
        break
        
