import json

SYSTEM_PROMPT = (
    "You are a legal risk analyst specializing in licensing agreements. "
    "Your task is to analyze a contract clause and assess its risk level from a specific party's perspective. "
    "Focus on the practical consequences for the party — what they stand to lose, what obligations they must perform, "
    "and how protected they are if things go wrong. As a rough guide: low risk implies little to no liability for the party "
    "and high risk implies exposure."
)

def format_example(clause_text: str, party: str, risk: str, explanation: str, clause_index: int) -> dict:
    user_content = (
        f"Analyze the following contract clause from the perspective of the **{party}**.\n\n"
        f"Clause:\n\"\"\"\n{clause_text}\n\"\"\"\n\n"
        f"Provide:\n1. Risk Level\n2. Explanation"
    )
    assistant_content = f"Risk Level: {risk}\nExplanation: {explanation}"

    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_content}"
        "<|eot_id|>"
    )

    return {
        "text": text,
        "_meta": {
            "clause_index": clause_index,
            "party": party,
            "risk": risk,
        }
    }

def convert(input_path: str, output_path: str):
    with open(input_path) as f:
        data = json.load(f)

    examples = []
    for record in data:
        if record.get("status") != "success":
            continue
        clause_text = record["clause_text"]
        clause_index = record["clause_index"]
        for party, info in record["parties"].items():
            examples.append(format_example(
                clause_text=clause_text,
                party=party,
                risk=info["risk"],
                explanation=info["explanation"],
                clause_index=clause_index,
            ))

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")

if __name__ == "__main__":
    convert("completed_annotation_merged.json", "finetune_merged.jsonl")