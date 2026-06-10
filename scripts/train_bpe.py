import json
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase

def train_tokenizer():
    dataset_path = 'data/synthetic_dataset.json'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run generate_dataset.py first.")
        return

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # NOTE: schema table/column names are deliberately NOT special tokens anymore.
    # They are emitted exclusively through the pointer/copy head as atomic IDs
    # appended after the BPE vocab (see prepare_fusion.py). If the generator could
    # spell them, the copy mechanism would never be forced to learn schema linking.

    # Write all text to a temporary file for HuggingFace trainer
    corpus_path = "data/temp_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(item["input"] + "\n")
            f.write(item["output"] + "\n")

    print("Training BPE Tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[unk]"))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    
    # We use a vocab size of 2000. It's enough to capture T-SQL keywords and common Amazon Logistics terms,
    # while forcing it to split rare typos into subwords.
    # [EOS] appended last so [PAD]=0 / [UNK]=1 ids stay stable; gives the SQL target an explicit stop token.
    trainer = BpeTrainer(special_tokens=["[pad]", "[unk]", "[cls]", "[sep]", "[eos]"], vocab_size=2000)
    
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    tokenizer_json_path = "data/bpe_tokenizer.json"
    tokenizer.save(tokenizer_json_path)
    
    # We also export the raw vocab and merges into simple text files 
    # so our custom C++ inference engine can easily load them without a JSON library!
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    
    vocab = t_data["model"]["vocab"]
    merges = t_data["model"]["merges"]
    
    # Sort vocab by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    with open("data/bpe_vocab.txt", "w", encoding="utf-8") as f:
        for word, _ in sorted_vocab:
            f.write(f"{word}\n")
            
    with open("data/bpe_merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge}\n")

    os.remove(corpus_path)
    print("BPE Training Complete!")
    print(f"Exported to data/bpe_vocab.txt ({len(vocab)} tokens)")
    print(f"Exported to data/bpe_merges.txt ({len(merges)} merges)")

if __name__ == "__main__":
    train_tokenizer()
