import json
import random

def oversample_licensor_high(file_path, num_to_add=100):
    entries = []
    licensor_high_entries = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            entries.append(data)
            meta = data.get('_meta', {})
            if meta.get('party') == 'Licensor' and meta.get('risk') == 'High':
                licensor_high_entries.append(data)
    
    if not licensor_high_entries:
        print("No Licensor - High Risk entries found to sample from!")
        return
        
    print(f"Current Licensor - High entries: {len(licensor_high_entries)}")
    
    # Sample 100 entries (with replacement if needed, but here we probably want to just pick 100 random ones)
    new_samples = random.choices(licensor_high_entries, k=num_to_add)
    
    print(f"Adding {len(new_samples)} sampled entries to {file_path}")
    
    # Append to the file
    with open(file_path, 'a') as f:
        for entry in new_samples:
            f.write(json.dumps(entry) + '\n')

oversample_licensor_high('train_balanced.jsonl', 100)
