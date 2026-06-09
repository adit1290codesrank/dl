import json
import re
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "..", "data", "synthetic_dataset.json")

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

string_regex = re.compile(r"'([^']+)'")
number_regex = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b") # Also catch unquoted dates just in case

fixed_count = 0
for sample in data:
    inp = sample["input"]
    out = sample["output"]
    
    literals = string_regex.findall(out)
    
    missing = []
    for lit in literals:
        if lit not in ["_CB_", "_CB_USERNAME_"] and lit.lower() not in inp.lower():
            missing.append(lit)
            
    if missing:
        sample["input"] = inp + " (using: " + ", ".join(missing) + ")"
        fixed_count += 1

with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Fixed {fixed_count} samples by appending missing SQL string literals to the English prompt.")
