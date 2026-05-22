import os
import numpy as np
import struct
import random
import re
from collections import Counter
from datasets import load_dataset

def augment(sentence, n=3):
    """Generate n augmented versions of a sentence"""
    words = sentence.split()
    augmented = []
    
    for _ in range(n):
        new_words = words.copy()
        r = random.random()
        
        if r < 0.33 and len(new_words) > 3:
            # Random deletion: drop a non-key word
            idx = random.randint(1, len(new_words)-2)
            new_words.pop(idx)
            
        elif r < 0.66 and len(new_words) > 2:
            # Random swap: swap two adjacent words
            idx = random.randint(0, len(new_words)-2)
            new_words[idx], new_words[idx+1] = new_words[idx+1], new_words[idx]
            
        else:
            # Random insertion of a filler word
            fillers = ["please", "can you", "just", "quickly", "hey", "now"]
            new_words.insert(random.randint(0,len(new_words)), random.choice(fillers))
        
        augmented.append(" ".join(new_words))
    
    return augmented

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def main():
    print("Loading FULL CLINC150 dataset from HuggingFace...")
    dataset = load_dataset("DeepPavlov/clinc_oos", "plus")
    features = dataset['train'].features
    
    # DeepPavlov/clinc_oos uses 'label' (int) and 'label_text' (string) instead of a ClassLabel
    if 'label' in features and 'label_text' in features:
        # Extract the mapping deterministically by iterating the dataset
        intent_map = {}
        for item in dataset['train']:
            intent_map[item['label']] = item['label_text']
            if len(intent_map) == 150: # Optimization: stop early once all 150 are found
                break
        
        # Sort by integer ID to ensure deterministic ordering
        intent_names = [intent_map[i] for i in sorted(intent_map.keys())]
        label_key = 'label'
    else:
        raise ValueError(f"Expected 'label' and 'label_text'. Available features: {list(features.keys())}")

    target_intents = intent_names

    target_ids = list(range(len(intent_names)))

    print("Building custom Corpus Vocabulary...")
    counter = Counter()
    train_data = []
    val_data = []

    # Extract target data and build vocab
    for item in dataset['train']:
        if item[label_key] in target_ids:
            train_data.append((item['text'], target_ids.index(item[label_key])))
            counter.update(tokenize(item['text']))

    for item in dataset['validation']:
        if item[label_key] in target_ids:
            val_data.append((item['text'], target_ids.index(item[label_key])))

    # Augment training data (3x)
    print(f"Extracted {len(train_data)} training examples.")
    augmented_train = []
    for text, label in train_data:
        augmented_train.append((text, label))
        for aug_text in augment(text, n=3):
            augmented_train.append((aug_text, label))
            counter.update(tokenize(aug_text))
    
    train_data = augmented_train
    print(f"Augmented train size: {len(train_data)}")

    # Keep tokens seen >= 1 times (include everything to reduce UNK hits)
    MIN_FREQ = 1
    vocab = ["[PAD]", "[UNK]"]
    vocab += [word for word, count in counter.most_common() if count >= MIN_FREQ]
    
    vocab_size = len(vocab)
    print(f"Custom Vocab Size: {vocab_size} (down from 30,522!)")

    word_to_id = {word: i for i, word in enumerate(vocab)}
    PAD_ID = word_to_id["[PAD]"]
    UNK_ID = word_to_id["[UNK]"]
    max_seq_len = 32

    def encode(data):
        tokens_list = []
        labels_list = []
        for text, label_idx in data:
            words = tokenize(text)
            ids = [word_to_id.get(w, UNK_ID) for w in words]
            # Pad or truncate
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            else:
                ids += [PAD_ID] * (max_seq_len - len(ids))
            
            tokens_list.append(ids)
            
            label = [0.0] * len(target_intents)
            label[label_idx] = 1.0
            labels_list.append(label)
            
        return np.array(tokens_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)

    print("Encoding sets with custom vocab...")
    train_tokens, train_labels = encode(train_data)
    test_tokens, test_labels = encode(val_data)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, "alexa.bin")
    
    print(f"Writing packed binary data to {output_file}...")
    with open(output_file, 'wb') as f:
        f.write(struct.pack('i', len(train_tokens)))
        f.write(struct.pack('i', len(test_tokens)))
        f.write(struct.pack('i', max_seq_len))
        f.write(struct.pack('i', vocab_size))
        f.write(struct.pack('i', len(target_intents)))
        
        f.write(train_tokens.tobytes())
        f.write(train_labels.tobytes())
        f.write(test_tokens.tobytes())
        f.write(test_labels.tobytes())

    print(f"Successfully created {output_file}!")

    # Export vocab and intents for the C++ interactive CLI
    vocab_file = os.path.join(data_dir, "vocab.txt")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write(f"{w}\n")
            
    intents_file = os.path.join(data_dir, "intents.txt")
    with open(intents_file, 'w', encoding='utf-8') as f:
        for name in target_intents:
            f.write(f"{name}\n")
            
    print(f"Exported vocab.txt and intents.txt for interactive CLI.")

if __name__ == "__main__":
    main()