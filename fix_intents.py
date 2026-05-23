from datasets import load_dataset
import os

def fix_intents():
    print("Downloading dataset metadata...")
    dataset = load_dataset("DeepPavlov/clinc_oos", "plus")
    
    # Extract all labels correctly without breaking early
    intent_map = {}
    for item in dataset['train']:
        intent_map[item['label']] = item['label_text']
        
    # The regular intents are EXACTLY IDs 0 to 149.
    # Out-of-scope (oos) is ID 150.
    target_intents = [intent_map[i] for i in range(150)]
    
    os.makedirs("data", exist_ok=True)
    intents_file = os.path.join("data", "intents.txt")
    
    with open(intents_file, 'w', encoding='utf-8') as f:
        for name in target_intents:
            f.write(f"{name}\n")
            
    print(f"Successfully fixed {intents_file}!")
    print(f"First 5 intents: {target_intents[:5]}")

if __name__ == "__main__":
    fix_intents()
