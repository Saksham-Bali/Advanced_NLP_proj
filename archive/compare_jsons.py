import json

def analyze(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading {filepath}: {e}"
    
    stats = {
        "total_clauses": len(data),
        "success_count": sum(1 for d in data if d.get("status") == "success"),
        "error_count": sum(1 for d in data if d.get("status") != "success"),
        "risk_dist": {"Licensor": {"High": 0, "Medium": 0, "Low": 0}, 
                      "Licensee": {"High": 0, "Medium": 0, "Low": 0}},
        "avg_explanation_len": 0,
        "has_trigger_quote": 0
    }
    
    total_len = 0
    exp_count = 0
    
    for item in data:
        if item.get("status") == "success" and "parties" in item:
            for party in ["Licensor", "Licensee"]:
                p_data = item["parties"].get(party)
                if p_data:
                    risk = p_data.get("risk", "Unknown")
                    if risk in stats["risk_dist"][party]:
                        stats["risk_dist"][party][risk] += 1
                    else:
                        stats["risk_dist"][party][risk] = 1
                    
                    if "explanation" in p_data:
                        total_len += len(p_data["explanation"])
                        exp_count += 1
                        
                    if "trigger_quote" in p_data or "evidence_quote" in p_data:
                        stats["has_trigger_quote"] += 1
                        
    if exp_count > 0:
        stats["avg_explanation_len"] = total_len / exp_count
        
    return stats

f1 = 'final_test1.json'
f2 = 'final_test_updated.json'

print(f"--- {f1} ---")
print(analyze(f1))
print(f"\n--- {f2} ---")
print(analyze(f2))

# Also let's compare some random clauses side by side to see quality
try:
    with open(f1, 'r') as f: d1 = json.load(f)
    with open(f2, 'r') as f: d2 = json.load(f)
    
    d1_dict = {item.get('clause_index'): item for item in d1 if 'clause_index' in item}
    d2_dict = {item.get('clause_index'): item for item in d2 if 'clause_index' in item}
    
    common_idx = list(set(d1_dict.keys()).intersection(set(d2_dict.keys())))
    if common_idx:
        print(f"\nComparing clause index {common_idx[0]}:")
        print("Text:", d1_dict[common_idx[0]].get("clause_text")[:100] + "...")
        print(f"[{f1}] Licensor: {d1_dict[common_idx[0]].get('parties', {}).get('Licensor')}")
        print(f"[{f2}] Licensor: {d2_dict[common_idx[0]].get('parties', {}).get('Licensor')}")
        
except Exception as e:
    print(e)
