import json

f1 = 'final_test1.json'
f2 = 'final_test_updated.json'

with open(f1, 'r') as f: d1 = json.load(f)
with open(f2, 'r') as f: d2 = json.load(f)

d1_dict = {item.get('clause_index'): item for item in d1 if 'clause_index' in item and item.get("status") == "success"}
d2_dict = {item.get('clause_index'): item for item in d2 if 'clause_index' in item and item.get("status") == "success"}

common_idx = sorted(list(set(d1_dict.keys()).intersection(set(d2_dict.keys()))))
mismatches = []

for idx in common_idx:
    c1 = d1_dict[idx]
    c2 = d2_dict[idx]
    
    r1_licensor = c1.get("parties", {}).get("Licensor", {}).get("risk")
    r1_licensee = c1.get("parties", {}).get("Licensee", {}).get("risk")
    
    r2_licensor = c2.get("parties", {}).get("Licensor", {}).get("risk")
    r2_licensee = c2.get("parties", {}).get("Licensee", {}).get("risk")
    
    if r1_licensor != r2_licensor or r1_licensee != r2_licensee:
        mismatches.append({
            "index": idx,
            "text": c1.get("clause_text"),
            "f1_licensor": r1_licensor, "f2_licensor": r2_licensor,
            "f1_licensee": r1_licensee, "f2_licensee": r2_licensee,
            "f1_exp_licensor": c1.get("parties", {}).get("Licensor", {}).get("explanation"),
            "f2_exp_licensor": c2.get("parties", {}).get("Licensor", {}).get("explanation"),
            "f1_exp_licensee": c1.get("parties", {}).get("Licensee", {}).get("explanation"),
            "f2_exp_licensee": c2.get("parties", {}).get("Licensee", {}).get("explanation"),
        })

print(f"Total Common Clauses: {len(common_idx)}")
print(f"Total Mismatches in Risk: {len(mismatches)}")

# Print a few mismatches to analyze quality
for i, m in enumerate(mismatches[:5]):
    print(f"\n--- Mismatch {i+1} (Clause {m['index']}) ---")
    print("Text:", m["text"])
    print(f"[final_test1.json]")
    print(f"  Licensor Risk: {m['f1_licensor']} | Exp: {m['f1_exp_licensor']}")
    print(f"  Licensee Risk: {m['f1_licensee']} | Exp: {m['f1_exp_licensee']}")
    print(f"[final_test_updated.json]")
    print(f"  Licensor Risk: {m['f2_licensor']} | Exp: {m['f2_exp_licensor']}")
    print(f"  Licensee Risk: {m['f2_licensee']} | Exp: {m['f2_exp_licensee']}")

