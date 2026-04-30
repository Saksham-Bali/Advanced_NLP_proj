import json
import random
import argparse
import re
import requests
from collections import defaultdict
import concurrent.futures

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm for the progress bar: pip install tqdm")
    exit(1)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b" # Found in your local ollama config

REPHRASE_PROMPT = """You are a legal editor. Rephrase the following explanation while preserving its EXACT meaning and technical nuance.
- Output ONLY the rephrased text. No conversational preamble. No quotation marks around it. No headers.
- Keep the length similar (1-2 sentences).
- Do NOT change the core logic or the facts.

Original Explanation: 
{explanation}
"""

def extract_explanation(text):
    match = re.search(r"Explanation:\s*(.*?)<\|eot_id\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def inject_explanation(text, new_explanation):
    return re.sub(
        r"(Explanation:\s*)(.*?)(<\|eot_id\|>)$", 
        rf"\g<1>{new_explanation}\g<3>", 
        text, 
        count=1,
        flags=re.DOTALL
    )

def get_ollama_rephrase(original_text):
    prompt = REPHRASE_PROMPT.format(explanation=original_text)
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json().get("response", "").strip()
            # Clean up the output in case the model ignored 'no quotes' instruction
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content
    except Exception as e:
        pass # Silently fail here, we track failures at the caller level
    return None

def process_item_copies(item, num_copies):
    original_text = item['text']
    base_expl = extract_explanation(original_text)
    new_items = []
    
    successes = 0
    failures = 0
    
    if not base_expl:
        for _ in range(num_copies):
            new_items.append(item.copy())
            failures += 1
        return new_items, successes, failures
        
    for _ in range(num_copies):
        rephrased = get_ollama_rephrase(base_expl)
        
        if not rephrased:
            new_items.append(item.copy())
            failures += 1
            continue
            
        new_item = item.copy()
        new_item['text'] = inject_explanation(original_text, rephrased)
        new_items.append(new_item)
        successes += 1
        
    return new_items, successes, failures

def balance_and_rephrase_ollama(input_file, output_file):
    
    # 1. Load data
    categories = defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            risk = data.get('_meta', {}).get('risk')
            if risk:
                categories[risk].append(data)
                
    if not categories:
        print("No valid data found.")
        return

    target_size = max(len(items) for items in categories.values())
    
    print("\n==================================")
    print(f"Connecting to local Ollama API (Model: {MODEL_NAME})...")
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
            
        needed = target_size - len(items)
        print(f"\n---> Enhancing '{risk}' category. Need {needed} synthetic variations.")
        
        base_copies_per_item = needed // len(items)
        remainder = needed % len(items)
        copy_counts = [base_copies_per_item] * len(items)
        for i in range(remainder):
            copy_counts[i] += 1
            
        balanced_data.extend(items)
        
        total_augmented = 0
        total_successes = 0
        total_failures = 0
        
        # Concurrency set to 2-3 to avoid crashing local memory if Ollama hits OOM
        # 3 workers usually maximizes GPU throughput locally
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for item, count in zip(items, copy_counts):
                if count > 0:
                    futures.append(executor.submit(process_item_copies, item, count))
            
            # Progress bar based on expected number of tasks
            with tqdm(total=len(futures), desc=f"Generating {risk} variations", unit="batch") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    new_augmented_items, s, f = future.result()
                    balanced_data.extend(new_augmented_items)
                    
                    total_augmented += len(new_augmented_items)
                    total_successes += s
                    total_failures += f
                    
                    # Update progress bar description dynamically
                    pbar.set_postfix(success=total_successes, err=total_failures)
                    pbar.update(1)
                    
        print(f"Finished '{risk}'! Successfully rephrased: {total_successes} | Exact duplicates used: {total_failures}")

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
    parser.add_argument("--output", "-o", type=str, default="finetune_rephrased_local.jsonl")
    args = parser.parse_args()
    
    balance_and_rephrase_ollama(args.input, args.output)
