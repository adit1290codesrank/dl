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
    
    print("Loading Jargon Dictionary...")
    jargon_dict_path = os.path.join(base_dir, "..", "jargon_dict.json")
    jargon_dict = {}
    if os.path.exists(jargon_dict_path):
        with open(jargon_dict_path, 'r', encoding='utf-8') as f:
            jargon_dict = json.load(f)
            
    print("Loading synthetic dataset...")
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
        
    print(f"Tokenizing {len(samples)} examples with BPE...")
    tokenized_samples = []
    for s in samples:
        inp = s["input"]
        for k, v in jargon_dict.items():
            inp = inp.replace(k, v) # deterministic stage 1 resolution
        out = s["output"]
        inp_ids = tokenizer.encode(inp).ids
        out_ids = tokenizer.encode(out).ids
        tokenized_samples.append((inp_ids, out_ids))
        
    schema_tokens_list = [tokenizer.encode(desc).ids for desc in schema_elements]
    
    n_train = int(len(tokenized_samples) * 0.9)
    n_val = len(tokenized_samples) - n_train
    seq_len = 256 # Increased to fit both English and SQL
    schema_size = len(schema_elements)
    
    train_samples = tokenized_samples[:n_train]
    val_samples = tokenized_samples[n_train:]
    
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None: pad_id = 0
    unk_id = tokenizer.token_to_id("[UNK]")
    if unk_id is None: unk_id = 1
    
    # We need a SEP token to divide English prompt from SQL target
    sep_id = tokenizer.token_to_id("\n")
    if sep_id is None: sep_id = 1

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
            combined = inp_ids + [sep_id] + out_ids
            x_seq = combined[:-1]
            y_seq = combined[1:]
            
            x_pad = pad_sequence(x_seq, seq_len)
            y_pad = pad_sequence(y_seq, seq_len)
            
            for j in range(seq_len):
                X[i * seq_len + j] = float(x_pad[j])
                Y[i * seq_len + j] = float(y_pad[j])
                
            for j in range(schema_size):
                Schema[i * schema_size + j] = float(schema_ids[j])
                
        X.tofile(f)
        Schema.tofile(f)
        Y.tofile(f)

    import hashlib
    def get_hash_vec(text, dim=2048):
        vec = [0.0] * dim
        text = text.lower()
        for i in range(len(text)-2):
            tg = text[i:i+3]
            idx = int(hashlib.md5(tg.encode()).hexdigest(), 16) % dim
            vec[idx] = 1.0
        for word in text.split():
            if len(word) > 2:
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
                vec[idx] = 1.0
        return vec
        
    print("Generating Frozen Lexical Keys...")
    K_frozen = array.array('f', [0.0] * (schema_size * 2048))
    for i, desc in enumerate(schema_elements):
        vec = get_hash_vec(desc)
        for j in range(2048):
            K_frozen[i * 2048 + j] = vec[j]

    out_bin = os.path.join(out_dir, "breakwalls.bin")
    with open(out_bin, "wb") as f:
        f.write(struct.pack("i", n_train))
        f.write(struct.pack("i", n_val))
        f.write(struct.pack("i", seq_len))
        f.write(struct.pack("i", vocab_size))
        f.write(struct.pack("i", schema_size))
        
        # Write global K_frozen
        K_frozen.tofile(f)
        
        write_set(f, train_samples, n_train)
        write_set(f, val_samples, n_val)
        
    with open(os.path.join(out_dir, "schema_strings.txt"), "w", encoding="utf-8") as f:
        for desc in schema_elements:
            f.write(desc.lower().replace('\n', ' ') + '\n')
            
    with open(os.path.join(out_dir, "jargon_dict.txt"), "w", encoding="utf-8") as f:
        for k, v in jargon_dict.items():
            f.write(f"{k}|{v}\n")
        
    print(f"Successfully generated {out_bin} from synthetic dataset!")
    print(f"Train samples: {n_train}, Val samples: {n_val}, Seq Len: {seq_len}, Schema Size: {schema_size}")

if __name__ == "__main__":
    generate_dataset()
