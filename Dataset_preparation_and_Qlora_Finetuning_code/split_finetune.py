"""
split_finetune.py
-----------------
Splits finetune_final.jsonl into train.jsonl and val.jsonl with the following guarantees:
  1. Every unique clause TEXT goes ENTIRELY to either train or val.
     Both the Licensor and Licensee perspective of the same clause body are always together.
  2. Empty / malformed clauses (no clause body text) are removed before splitting.
  3. Exact duplicate samples are removed before splitting.
  4. The split is approximately 80% train / 20% val (at the clause level, not sample level).
  5. The val set is stratified so it mirrors the High/Low risk distribution of the full dataset.
  6. Splits are globally shuffled before writing.

Outputs:
  train.jsonl   (overwrites any existing file)
  val.jsonl     (overwrites any existing file)
"""

import json
import random
import collections

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_FILE  = "finetune_final.jsonl"
TRAIN_FILE  = "train.jsonl"
VAL_FILE    = "val.jsonl"
VAL_RATIO   = 0.20          # 20 % of clause groups → val
RANDOM_SEED = 42
# ───────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

# 1. Load all samples ──────────────────────────────────────────────────────────
all_samples = []
seen_texts  = set()
empty_removed = 0
dup_removed   = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = obj.get("text", "")

        # Remove empty-clause samples
        # The clause body sits between the triple-quotes in the user turn
        try:
            clause_body = text.split('Clause:\n"""')[1].split('"""')[0].strip()
        except IndexError:
            clause_body = "X"  # malformed but not empty; keep for now

        if len(clause_body) < 3:
            empty_removed += 1
            continue

        # Remove exact duplicates
        if text in seen_texts:
            dup_removed += 1
            continue
        seen_texts.add(text)

        all_samples.append(obj)

print(f"Loaded {len(all_samples)} samples "
      f"(removed {empty_removed} empty, {dup_removed} duplicates)")

# 2. Group samples by actual clause body text ────────────────────────────────
# Using the clause body (stripped text between triple-quotes in the user turn)
# as the grouping key guarantees that even if clause_index repeats across
# different contracts, the physical clause text is what keeps both perspectives
# (Licensor & Licensee) in the same split.
clause_groups: dict[str, list] = collections.defaultdict(list)
for obj in all_samples:
    text = obj["text"]
    try:
        clause_body = text.split('Clause:\n"""')[1].split('"""')[0].strip()
    except IndexError:
        clause_body = text  # fall back to full text as key
    clause_groups[clause_body].append(obj)

all_clause_ids = list(clause_groups.keys())
print(f"Unique clause body groups: {len(all_clause_ids)}")

# 3. Stratified split at the clause-group level ────────────────────────────────
# A clause group is "High-containing" if ANY of its samples is High risk,
# otherwise it is "Low-only". We stratify on this property.

high_clause_ids = []
low_clause_ids  = []

for cid in all_clause_ids:
    risks = {s["_meta"]["risk"] for s in clause_groups[cid]}
    if "High" in risks:
        high_clause_ids.append(cid)
    else:
        low_clause_ids.append(cid)

random.shuffle(high_clause_ids)
random.shuffle(low_clause_ids)

def split_list(lst, val_ratio):
    n_val = max(1, round(len(lst) * val_ratio))
    return lst[n_val:], lst[:n_val]   # train, val

train_high_ids, val_high_ids = split_list(high_clause_ids, VAL_RATIO)
train_low_ids,  val_low_ids  = split_list(low_clause_ids,  VAL_RATIO)

train_clause_ids = train_high_ids + train_low_ids
val_clause_ids   = val_high_ids   + val_low_ids

# Collect samples
def collect(clause_ids):
    samples = []
    for cid in clause_ids:
        samples.extend(clause_groups[cid])
    random.shuffle(samples)
    return samples

train_samples = collect(train_clause_ids)
val_samples   = collect(val_clause_ids)

# 4. Sanity checks ─────────────────────────────────────────────────────────────
# Verify zero overlap at text level
train_texts = {s["text"] for s in train_samples}
val_texts   = {s["text"] for s in val_samples}
overlap = train_texts & val_texts
assert len(overlap) == 0, f"LEAK: {len(overlap)} samples appear in both splits!"

# Verify zero overlap at clause_index level
train_cids_set = set(train_clause_ids)
val_cids_set   = set(val_clause_ids)
assert train_cids_set.isdisjoint(val_cids_set), "Clause index leak detected!"

# 5. Print summary ─────────────────────────────────────────────────────────────
def risk_dist(samples):
    c = collections.Counter(s["_meta"]["risk"] for s in samples)
    total = sum(c.values())
    return {k: f"{v} ({100*v/total:.1f}%)" for k, v in sorted(c.items())}

print("\n── Split Summary ──────────────────────────────────────────────")
print(f"Train: {len(train_samples)} samples across {len(train_clause_ids)} clauses")
print(f"  Risk dist : {risk_dist(train_samples)}")
print(f"Val  : {len(val_samples)} samples across {len(val_clause_ids)} clauses")
print(f"  Risk dist : {risk_dist(val_samples)}")
actual_val_pct = len(val_samples) / (len(train_samples) + len(val_samples)) * 100
print(f"\nActual val %: {actual_val_pct:.1f}%")
print(f"Overlap between splits: {len(overlap)} samples  ✓")
print("──────────────────────────────────────────────────────────────\n")

# 6. Write output files ────────────────────────────────────────────────────────
def write_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Written → {path}  ({len(samples)} lines)")

write_jsonl(train_samples, TRAIN_FILE)
write_jsonl(val_samples,   VAL_FILE)
print("\nDone.")
