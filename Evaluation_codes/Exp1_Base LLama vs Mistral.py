# ============================================================
# Zero-Shot Baseline: Legal Contract Risk Classification
# Runs both LLaMA 3.1 8B and Mistral 7B Instruct via Ollama
# ============================================================
# SETUP: Run these in terminal before running this script:
#   1. Download Ollama: https://ollama.com
#   2. ollama pull llama3.1:8b
#   3. ollama pull mistral:7b-instruct
#   4. pip install ollama pandas scikit-learn
# ============================================================

import ollama
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# STEP 1 — Load Dataset
# ============================================================
df = pd.read_csv("legal_contract_clauses.csv")  

# Adjust these to match your actual column names
CLAUSE_COL = "clause_text"   
TYPE_COL   = "clause_type"   
RISK_COL   = "risk_level"    

print("Dataset shape:", df.shape)
print("\nSample row:")
print(df.head(2))
print("\nRisk level distribution:")
print(df[RISK_COL].value_counts())

# ============================================================
# STEP 2 — Sample Dataset
# Both models run on the SAME sample for fair comparison
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
# Same prompt used for both models — keeps comparison fair
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
# STEP 4 — Inference Function
# Takes model_name as parameter so same function runs both models
# ============================================================
def get_prediction(clause_text: str, clause_type: str, model_name: str) -> str:
    prompt = build_prompt(clause_text, clause_type)

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,    # greedy decoding — deterministic output
            "num_predict": 10,   # we only need one word
        }
    )

    generated = response["message"]["content"].strip()

    # Extract label — look for Low/Medium/High in response
    match = re.search(r"\b(Low|Medium|High)\b", generated, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # Fallback: sometimes model says "the risk level is high"
    generated_lower = generated.lower()
    if "high" in generated_lower:
        return "High"
    elif "medium" in generated_lower:
        return "Medium"
    elif "low" in generated_lower:
        return "Low"

    return "Unknown"

# ============================================================
# STEP 5 — Run Inference for a Single Model
# Returns accuracy, f1, and saves results CSV
# ============================================================
def run_evaluation(model_name: str, model_label: str) -> dict:
    print(f"\n{'='*50}")
    print(f"Running: {model_label}")
    print(f"{'='*50}")

    predictions = []
    actuals     = []
    failed      = 0

    for i, row in df_sample.iterrows():
        clause      = str(row[CLAUSE_COL])
        clause_type = str(row[TYPE_COL])
        actual      = str(row[RISK_COL]).strip().capitalize()

        pred = get_prediction(clause, clause_type, model_name)

        predictions.append(pred)
        actuals.append(actual)

        if pred == "Unknown":
            failed += 1

        if len(predictions) % 10 == 0:
            correct_so_far = sum(p == a for p, a in zip(predictions, actuals))
            running_acc = correct_so_far / len(predictions)
            print(f"  [{len(predictions)}/{len(df_sample)}] Running accuracy: {running_acc*100:.1f}%")

    print(f"Done. Failed to parse: {failed}/{len(df_sample)}")

    # Filter unknowns for clean metrics
    valid_mask    = [p != "Unknown" for p in predictions]
    preds_clean   = [p for p, v in zip(predictions, valid_mask) if v]
    actuals_clean = [a for a, v in zip(actuals,     valid_mask) if v]

    labels    = ["Low", "Medium", "High"]
    accuracy  = accuracy_score(actuals_clean, preds_clean)
    f1_macro  = f1_score(actuals_clean, preds_clean, average="macro",    labels=labels, zero_division=0)
    f1_weighted = f1_score(actuals_clean, preds_clean, average="weighted", labels=labels, zero_division=0)

    # Per-class breakdown
    print(f"\nAccuracy:    {accuracy*100:.1f}%")
    print(f"Macro F1:    {f1_macro:.3f}")
    print(f"Weighted F1: {f1_weighted:.3f}")
    print("\nPer-class breakdown:")
    print(classification_report(actuals_clean, preds_clean, labels=labels, zero_division=0))

    # Error analysis
    results_df = df_sample.copy().reset_index(drop=True)
    results_df["predicted_risk"] = predictions
    results_df["correct"]        = [p == a for p, a in zip(predictions, actuals)]

    errors_df = results_df[results_df["correct"] == False]
    print(f"Total errors: {len(errors_df)}")
    print("Confusion breakdown (Actual → Predicted):")
    confusion = errors_df.groupby([RISK_COL, "predicted_risk"]).size().reset_index(name="count")
    print(confusion.to_string(index=False))


    return {
        "label":       model_label,
        "accuracy":    accuracy,
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
        "failed":      failed,
        "total":       len(preds_clean),
    }

# ============================================================
# STEP 6 — Run Both Models
# Both run on the exact same df_sample for fair comparison
# ============================================================
results = []

results.append(run_evaluation("llama3.1:8b",          "Zero-shot LLaMA 3.1 8B"))
results.append(run_evaluation("mistral:7b-instruct",   "Zero-shot Mistral 7B Instruct"))

# ============================================================
# STEP 7 — Final Comparison Table
# ============================================================
print("\n")
print("=" * 60)
print("FINAL COMPARISON TABLE")
print("=" * 60)
print(f"| Model                               | Accuracy | Macro F1 |")
print(f"|-------------------------------------|----------|----------|")
for r in results:
    print(f"| {r['label']:<35} | {r['accuracy']*100:>6.1f}%  | {r['f1_macro']:>8.3f} |")
print(f"| {'Fine-tuned LLaMA (no role)':<35} | {'TBD':>8} | {'TBD':>8} |")
print(f"| {'Fine-tuned Mistral (no role)':<35} | {'TBD':>8} | {'TBD':>8} |")
print(f"| {'Ours — best model (role-conditioned)':<35} | {'TBD':>8} | {'TBD':>8} |")