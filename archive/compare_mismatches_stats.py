import json

f1 = 'final_test1.json'
f2 = 'final_test_updated.json'

with open(f1, 'r') as f: d1 = json.load(f)
with open(f2, 'r') as f: d2 = json.load(f)

d1_dict = {item.get('clause_index'): item for item in d1 if 'clause_index' in item and item.get("status") == "success"}
d2_dict = {item.get('clause_index'): item for item in d2 if 'clause_index' in item and item.get("status") == "success"}

common_idx = sorted(list(set(d1_dict.keys()).intersection(set(d2_dict.keys()))))

transitions = {}
for idx in common_idx:
    c1 = d1_dict[idx]
    c2 = d2_dict[idx]
    
    for party in ["Licensor", "Licensee"]:
        r1 = c1.get("parties", {}).get(party, {}).get("risk", "Unknown")
        r2 = c2.get("parties", {}).get(party, {}).get("risk", "Unknown")
        
        if r1 != r2:
            trans = f"[{party}] {r1} -> {r2}"
            transitions[trans] = transitions.get(trans, 0) + 1

for t, c in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
    print(f"{t}: {c}")
