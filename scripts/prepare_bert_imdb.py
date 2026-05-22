import os
import numpy as np
import struct
from datasets import load_dataset
from transformers import BertTokenizer

def main():
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    max_seq_len = 128 # Kept small (128) to train fast on our custom CUDA engine

    def encode_dataset(data, seq_len):
        tokens = []
        labels = []
        for i, item in enumerate(data):
            encoded = tokenizer.encode(item['text'], add_special_tokens=True, max_length=seq_len, padding='max_length', truncation=True)
            tokens.append(encoded)
            # One-hot label (0 = negative, 1 = positive)
            label = [0.0, 0.0]
            label[item['label']] = 1.0
            labels.append(label)
            if i % 5000 == 0 and i > 0:
                print(f"  Processed {i} examples...")
        return np.array(tokens, dtype=np.float32), np.array(labels, dtype=np.float32)

    print("Encoding Training set (25,000 reviews)...")
    train_tokens, train_labels = encode_dataset(dataset['train'], max_seq_len)

    print("Encoding Validation set (25,000 reviews)...")
    test_tokens, test_labels = encode_dataset(dataset['test'], max_seq_len)

    print(f"Train shapes: tokens {train_tokens.shape}, labels {train_labels.shape}")
    print(f"Test shapes: tokens {test_tokens.shape}, labels {test_labels.shape}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, "imdb.bin")
    
    print(f"Writing packed binary data to {output_file}...")
    with open(output_file, 'wb') as f:
        # Header
        f.write(struct.pack('i', len(train_tokens)))
        f.write(struct.pack('i', len(test_tokens)))
        f.write(struct.pack('i', max_seq_len))
        f.write(struct.pack('i', vocab_size))
        
        # Payloads
        f.write(train_tokens.tobytes())
        f.write(train_labels.tobytes())
        f.write(test_tokens.tobytes())
        f.write(test_labels.tobytes())

    print(f"Successfully created {output_file} for the C++ CUDA engine!")

if __name__ == "__main__":
    main()
