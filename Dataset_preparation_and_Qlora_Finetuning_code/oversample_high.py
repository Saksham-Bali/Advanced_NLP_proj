"""
oversample_high.py
------------------
Reads train.jsonl, oversamples High-risk clauses with replacement until the
Low:High ratio reaches ~1.5:1, then writes the shuffled result to train_balanced.jsonl.

  Current:   903 High  / 2385 Low  →  ratio ~2.64:1
  Target:   1590 High  / 2385 Low  →  ratio ~1.50:1   (+687 synthetic High samples)
"""

import json
import random
import collections

INPUT_FILE  = "train.jsonl"
OUTPUT_FILE = "train_balanced.jsonl"
TARGET_RATIO = 1.5   # Low : High
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)

# ── 1. Load ──────────────────────────────────────────────────────────────────
all_samples  = []
high_samples = []
low_samples  = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        all_samples.append(obj)
        risk = obj["_meta"]["risk"]
        if risk == "High":
            high_samples.append(obj)
        else:
            low_samples.append(obj)

n_high = len(high_samples)
n_low  = len(low_samples)
print(f"Original → High: {n_high}, Low: {n_low}  (ratio {n_low/n_high:.2f}:1)")

# ── 2. Compute how many extra High samples we need ───────────────────────────
# We want: n_low / (n_high + extra) = TARGET_RATIO
# => extra = n_low / TARGET_RATIO - n_high
target_n_high = round(n_low / TARGET_RATIO)
n_extra       = max(0, target_n_high - n_high)
print(f"Target  → High: {target_n_high}, Low: {n_low}  (ratio {n_low/target_n_high:.2f}:1)")
print(f"Sampling {n_extra} additional High examples with replacement ...")

# ── 3. Oversample with replacement ───────────────────────────────────────────
extra_samples = random.choices(high_samples, k=n_extra)

# ── 4. Combine and shuffle ───────────────────────────────────────────────────
balanced = all_samples + extra_samples
random.shuffle(balanced)

# ── 5. Verify final distribution ─────────────────────────────────────────────
final_risk = collections.Counter(s["_meta"]["risk"] for s in balanced)
final_high = final_risk["High"]
final_low  = final_risk["Low"]
print(f"\n── Final Distribution ─────────────────────────────────────────")
print(f"  Total samples : {len(balanced)}")
print(f"  High          : {final_high} ({100*final_high/len(balanced):.1f}%)")
print(f"  Low           : {final_low}  ({100*final_low/len(balanced):.1f}%)")
print(f"  Low:High ratio: {final_low/final_high:.2f}:1")
print(f"───────────────────────────────────────────────────────────────\n")

# ── 6. Write output ───────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in balanced:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Written → {OUTPUT_FILE}  ({len(balanced)} lines)")
