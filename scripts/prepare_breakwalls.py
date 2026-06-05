import os
import json
import struct
import array
from tokenizers import Tokenizer

def load_schema(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    schema_elements = []
    for item in tables:
        if item.get("INCLUDE_IN_MODEL", False):
            col_name = item.get("COLUMN_NAME", "")
            table_name = item.get("TABLE_NAME", "")
            synonyms = item.get("ALT_SYNONYMS", "")
            desc = f"{table_name} {col_name} {synonyms if synonyms else ''}"
            schema_elements.append(desc)
            
    return schema_elements

def generate_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tables_path = os.path.join(base_dir, "..", "all_tables.json")
    samples_path = os.path.join(base_dir, "..", "data", "synthetic_dataset.json")
    out_dir = os.path.join(base_dir, "..", "data")
    
    tokenizer_path = os.path.join(out_dir, "bpe_tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("Error: Run train_bpe.py first!")
        return
        
    print("Loading HuggingFace Tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"BPE Vocab Size: {vocab_size}")
    
    print("Loading schema...")
    schema_elements = load_schema(tables_path)
    
    print("Loading synthetic dataset...")
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
        
    print(f"Tokenizing {len(samples)} examples with BPE...")
    tokenized_samples = []
    for s in samples:
        inp = s["input"]
        out = s["output"]
        inp_ids = tokenizer.encode(inp).ids
        out_ids = tokenizer.encode(out).ids
        tokenized_samples.append((inp_ids, out_ids))
        
    schema_tokens_list = [tokenizer.encode(desc).ids for desc in schema_elements]
    
    n_train = int(len(tokenized_samples) * 0.9)
    n_val = len(tokenized_samples) - n_train
    seq_len = 64
    schema_size = len(schema_elements)
    
    train_samples = tokenized_samples[:n_train]
    val_samples = tokenized_samples[n_train:]
    
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None: pad_id = 0
    unk_id = tokenizer.token_to_id("[UNK]")
    if unk_id is None: unk_id = 1

    def pad_sequence(ids, length):
        if len(ids) > length:
            return ids[:length]
        return ids + [pad_id] * (length - len(ids))

    def write_set(f, dataset, n):
        X = array.array('f', [0.0] * (n * seq_len))
        Y = array.array('f', [0.0] * (n * seq_len))
        Schema = array.array('f', [0.0] * (n * schema_size))
        
        schema_ids = [toks[0] if toks else unk_id for toks in schema_tokens_list]
        
        for i, (inp_ids, out_ids) in enumerate(dataset):
            inp_pad = pad_sequence(inp_ids, seq_len)
            out_pad = pad_sequence(out_ids, seq_len)
            
            for j in range(seq_len):
                X[i * seq_len + j] = float(inp_pad[j])
                Y[i * seq_len + j] = float(out_pad[j])
                
            for j in range(schema_size):
                Schema[i * schema_size + j] = float(schema_ids[j])
                
        X.tofile(f)
        Schema.tofile(f)
        Y.tofile(f)

    out_bin = os.path.join(out_dir, "breakwalls.bin")
    with open(out_bin, "wb") as f:
        f.write(struct.pack("i", n_train))
        f.write(struct.pack("i", n_val))
        f.write(struct.pack("i", seq_len))
        f.write(struct.pack("i", vocab_size))
        f.write(struct.pack("i", schema_size))
        
        write_set(f, train_samples, n_train)
        write_set(f, val_samples, n_val)
        
    print(f"Successfully generated {out_bin} from synthetic dataset!")
    print(f"Train samples: {n_train}, Val samples: {n_val}, Seq Len: {seq_len}, Schema Size: {schema_size}")

if __name__ == "__main__":
    generate_dataset()
