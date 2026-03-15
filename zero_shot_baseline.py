# ============================================================
# Zero-Shot Baseline: Legal Contract Risk Classification
# Runs locally on Mac using Ollama — no GPU or HF token needed
# ============================================================
# SETUP: Run these in terminal before running this script:
#   1. Download Ollama: https://ollama.com
#   2. ollama pull llama3.1:8b
#   3. pip install ollama pandas scikit-learn
# ============================================================

import ollama
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# STEP 1 — Load Dataset
# ============================================================
df = pd.read_csv("legal_contract_clauses.csv")  # <-- change to your filename

# Adjust these to match your actual column names
CLAUSE_COL = "clause_text"   # <-- change if different
TYPE_COL   = "clause_type"   # <-- change if different
RISK_COL   = "risk_level"    # <-- change if different

print("Dataset shape:", df.shape)
print("\nSample row:")
print(df.head(2))
print("\nRisk level distribution:")
print(df[RISK_COL].value_counts())

# ============================================================
# STEP 2 — Sample Dataset
# On Mac with Ollama, you can comfortably run 200-300 samples
# 8B model runs at ~5-10 tokens/sec on Apple Silicon (M1/M2/M3)
# ============================================================
SAMPLE_SIZE = 200

# Stratified sample — equal representation per risk level
df_sample = (
    df.groupby(RISK_COL, group_keys=False)
    .apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE // 3), random_state=42))
    .reset_index(drop=True)
)

print(f"\nSampled {len(df_sample)} rows")
print(df_sample[RISK_COL].value_counts())

# ============================================================
# STEP 3 — Prompt Builder
# Zero-shot: no examples, just clear instructions
# ============================================================
def build_prompt(clause_text: str, clause_type: str) -> str:
    return f"""You are a legal risk analyst. Your task is to classify the risk level of a legal contract clause.

Clause Type: {clause_type}
Clause Text: {clause_text}

Classify the risk level of this clause as exactly one of: Low, Medium, High
- Low: Clause is standard and poses minimal risk to either party
- Medium: Clause may require negotiation or careful attention
- High: Clause poses significant legal or financial risk

Respond with only one word — Low, Medium, or High. Do not explain.

Risk Level:"""

# ============================================================
# STEP 4 — Inference Function using Ollama
# Ollama runs as a local server and exposes a simple Python API
# ============================================================
def get_prediction(clause_text: str, clause_type: str) -> str:
    prompt = build_prompt(clause_text, clause_type)

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,        # greedy decoding — deterministic output
            "num_predict": 10,       # we only need one word
        }
    )

    generated = response["message"]["content"].strip()

    # Extract label — look for Low/Medium/High in response
    match = re.search(r"\b(Low|Medium|High)\b", generated, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # Sometimes model says "risk level is high" — catch that too
    generated_lower = generated.lower()
    if "high" in generated_lower:
        return "High"
    elif "medium" in generated_lower:
        return "Medium"
    elif "low" in generated_lower:
        return "Low"

    return "Unknown"  # fallback

# ============================================================
# STEP 5 — Run Inference
# ============================================================
print(f"\nRunning zero-shot inference on {len(df_sample)} clauses...")
print("On Apple Silicon (M1/M2/M3) expect ~2-3 sec per clause")
print("On Intel Mac expect ~5-8 sec per clause\n")

predictions = []
actuals     = []
failed      = 0

for i, row in df_sample.iterrows():
    clause      = str(row[CLAUSE_COL])
    clause_type = str(row[TYPE_COL])
    actual      = str(row[RISK_COL]).strip().capitalize()

    pred = get_prediction(clause, clause_type)

    predictions.append(pred)
    actuals.append(actual)

    if pred == "Unknown":
        failed += 1

    # Progress update every 10 rows
    if len(predictions) % 10 == 0:
        correct_so_far = sum(p == a for p, a in zip(predictions, actuals))
        running_acc = correct_so_far / len(predictions)
        print(f"  [{len(predictions)}/{len(df_sample)}] Running accuracy: {running_acc*100:.1f}%")

print(f"\nInference complete. Failed to parse: {failed}/{len(df_sample)}")

# ============================================================
# STEP 6 — Evaluate Results
# ============================================================

# Filter out Unknown predictions
valid_mask    = [p != "Unknown" for p in predictions]
preds_clean   = [p for p, v in zip(predictions, valid_mask) if v]
actuals_clean = [a for a, v in zip(actuals,     valid_mask) if v]

labels = ["Low", "Medium", "High"]

accuracy    = accuracy_score(actuals_clean, preds_clean)
f1_macro    = f1_score(actuals_clean, preds_clean, average="macro",    labels=labels, zero_division=0)
f1_weighted = f1_score(actuals_clean, preds_clean, average="weighted", labels=labels, zero_division=0)

print("\n" + "="*50)
print("ZERO-SHOT BASELINE RESULTS")
print("="*50)
print(f"Model:              llama3.1:8b (via Ollama)")
print(f"Samples evaluated:  {len(preds_clean)}/{len(df_sample)}")
print(f"Failed to parse:    {failed}")
print(f"Accuracy:           {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"Macro F1:           {f1_macro:.4f}")
print(f"Weighted F1:        {f1_weighted:.4f}")
print("\nPer-class breakdown:")
print(classification_report(actuals_clean, preds_clean, labels=labels, zero_division=0))

# ============================================================
# STEP 7 — Save Results
# ============================================================
results_df = df_sample.copy().reset_index(drop=True)
results_df["predicted_risk"] = predictions
results_df["correct"]        = [p == a for p, a in zip(predictions, actuals)]

results_df.to_csv("zero_shot_results.csv", index=False)
print("Results saved to zero_shot_results.csv")

# ============================================================
# STEP 8 — Error Analysis
# Shows where the model is getting confused — useful for report
# ============================================================
print("\n--- ERROR ANALYSIS ---")
errors_df = results_df[results_df["correct"] == False]
print(f"Total errors: {len(errors_df)}")
print("\nConfusion breakdown (Actual → Predicted):")
confusion = errors_df.groupby([RISK_COL, "predicted_risk"]).size().reset_index(name="count")
print(confusion.to_string(index=False))

# ============================================================
# STEP 9 — Report Table
# ============================================================
print(f"| Model                   | Accuracy | Macro F1 |")
print(f"|-------------------------|----------|----------|")
print(f"| Zero-shot LLaMA 3.1 8B  | {accuracy*100:.1f}%    | {f1_macro:.3f}    |")
print(f"| Ours (role-conditioned) | TBD      | TBD      |")