import json

filtered_clauses = []

with open("complete.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        
        if (
            data.get("parties", {}).get("Licensor", {}).get("risk") == "Low" and
            data.get("parties", {}).get("Licensee", {}).get("risk") == "Low"
        ):
            filtered_clauses.append(data)

# Write to a text file in a readable format
with open("low_risk_clauses.txt", "w", encoding="utf-8") as out:
    for i, clause in enumerate(filtered_clauses, 1):
        out.write(f"Clause {i}\n")
        out.write("-" * 60 + "\n")
        out.write(f"Text: {clause.get('clause_text', 'N/A')}\n\n")
        
        out.write("Licensor:\n")
        out.write(f"  Risk: {clause['parties']['Licensor']['risk']}\n")
        out.write(f"  Explanation: {clause['parties']['Licensor']['explanation']}\n\n")
        
        out.write("Licensee:\n")
        out.write(f"  Risk: {clause['parties']['Licensee']['risk']}\n")
        out.write(f"  Explanation: {clause['parties']['Licensee']['explanation']}\n")
        
        out.write("\n" + "=" * 60 + "\n\n")

print("Filtered clauses saved to low_risk_clauses.txt")