import json
import random
import argparse
import os
import re
import time
from collections import defaultdict
import concurrent.futures

try:
    from groq import Groq
except ImportError:
    print("Please install groq: pip install groq")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    exit(1)

# Prompt to rephrase
REPHRASE_PROMPT = """You are a legal editor. Your task is to rephrase the following legal explanation while preserving its EXACT meaning, risk assessment, and technical nuance.
- Keep the length similar (1-2 sentences).
- Do NOT change the core logic or the facts.
- Do NOT include any preamble, headers, or <scratchpad>. Just return the rephrased text.

Original Explanation: 
"{explanation}"
"""

def extract_explanation(text):
    """Extracts the explanation part from the Llama chat formatted text."""
    match = re.search(r"Explanation:\s*(.*?)<\|eot_id\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def inject_explanation(text, new_explanation):
    """Replaces the explanation part in the Llama chat formatted text."""
    return re.sub(
        r"(Explanation:\s*)(.*?)(<\|eot_id\|>)$", 
        rf"\g<1>{new_explanation}\g<3>", 
        text, 
        count=1,
        flags=re.DOTALL
    )

def get_rephrased_explanation(client, original_text, attempt=1):
    """Calls Groq to get a single rephrased explanation."""
    prompt = REPHRASE_PROMPT.format(explanation=original_text)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.8, # slightly higher for varied phrasing
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        # strip quotes if the model wrapped it
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        return content

    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            wait = 15 * attempt
            # Simple backoff silently
            time.sleep(wait)
            if attempt < 5:
                return get_rephrased_explanation(client, original_text, attempt + 1)
        return None

def process_item_copies(client, item, num_copies):
    """Creates N copies of an item, rephrasing the explanation for each."""
    original_text = item['text']
    base_expl = extract_explanation(original_text)
    
    new_items = []
    successes = 0
    failures = 0
    
    # If we couldn't parse the explanation, just duplicate exactly
    if not base_expl:
        for _ in range(num_copies):
            new_items.append(item.copy())
            failures += 1
        return new_items, successes, failures
        
    for _ in range(num_copies):
        rephrased = get_rephrased_explanation(client, base_expl)
        
        # Fall back to original if API fails entirely
        if not rephrased:
            new_items.append(item.copy())
            failures += 1
            continue
            
        new_item = item.copy()
        new_item['text'] = inject_explanation(original_text, rephrased)
        new_items.append(new_item)
        successes += 1
        
    return new_items, successes, failures

def balance_and_rephrase(input_file, output_file, groq_key):
    try:
        client = Groq(api_key=groq_key)
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        return

    # 1. Load data
    categories = defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            risk = data.get('_meta', {}).get('risk')
            if risk:
                categories[risk].append(data)

    target_size = max(len(items) for items in categories.values())
    print("\n==================================")
    print("Connecting to Groq API (Model: llama-3.3-70b-versatile)...")
    print("==================================")
    print("--- Current Database Distribution ---")
    for risk, items in categories.items():
        print(f"[{risk.upper()}]: {len(items)} items")
    print(f"Target size per class is: {target_size}\n")
    
    balanced_data = []
    
    for risk, items in categories.items():
        if len(items) == target_size:
            balanced_data.extend(items)
            continue
            
        # We need to add (target_size - current) items
        needed = target_size - len(items)
        print(f"Enhancing '{risk}' category. Need {needed} synthetic variations.")
        
        # Distribute the needed copies across the existing items evenly
        base_copies_per_item = needed // len(items)
        remainder = needed % len(items)
        
        # Build list of exact copy counts per item
        copy_counts = [base_copies_per_item] * len(items)
        for i in range(remainder):
            copy_counts[i] += 1
            
        # Add original items
        balanced_data.extend(items)
        
        total_augmented = 0
        total_successes = 0
        total_failures = 0
        
        # Use concurrent futures to parallelize API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for item, count in zip(items, copy_counts):
                if count > 0:
                    futures.append(executor.submit(process_item_copies, client, item, count))
            
            with tqdm(total=len(futures), desc=f"Generating {risk} variations", unit="batch") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    new_augmented_items, s, f = future.result()
                    balanced_data.extend(new_augmented_items)
                    
                    total_augmented += len(new_augmented_items)
                    total_successes += s
                    total_failures += f
                    
                    pbar.set_postfix(success=total_successes, err=total_failures)
                    pbar.update(1)

        print(f"Finished '{risk}'! Successfully rephrased: {total_successes} | Exact duplicates used: {total_failures}\n")

    print("\nShuffling combined dataset to prevent clustering...")
    random.shuffle(balanced_data)

    with open(output_file, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item) + '\n')

    print("\n==================================")
    print("--- Final Balanced Distribution ---")
    counts = defaultdict(int)
    for item in balanced_data:
        counts[item['_meta']['risk']] += 1
    for risk, count in counts.items():
        print(f"[{risk.upper()}]: {count}")
    print(f"\nTotal examples ready for training: {len(balanced_data)}")
    print(f"File Saved to: {output_file}")
    print("==================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="finetune.jsonl")
    parser.add_argument("--output", "-o", type=str, default="finetune_rephrased.jsonl")
    parser.add_argument("--groq-key", type=str, default=os.getenv("GROQ_API_KEY"))
    args = parser.parse_args()
    
    if not args.groq_key:
        print("Please set the GROQ_API_KEY environment variable or pass --groq-key YOUR_KEY")
        exit(1)
        
    balance_and_rephrase(args.input, args.output, args.groq_key)
