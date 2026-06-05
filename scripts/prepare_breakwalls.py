import os
import json
import struct
import array
import re

def tokenize(text):
    # Simple tokenizer: lowercase, separate punctuation
    text = str(text).lower()
    text = re.sub(r"([.,!?()'\"\[\]\{\}])", r" \1 ", text)
    return [w for w in text.split() if w.strip()]

def load_schema(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    schema_elements = []
    for item in tables:
        if item.get("INCLUDE_IN_MODEL", False):
            col_name = item.get("COLUMN_NAME", "")
            table_name = item.get("TABLE_NAME", "")
            synonyms = item.get("ALT_SYNONYMS", "")
            
            # We construct a descriptive string for each included column
            desc = f"{table_name} {col_name} {synonyms if synonyms else ''}"
            schema_elements.append(desc)
            
    return schema_elements

def load_samples(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    pairs = []
    for s in samples:
        input_text = s.get("input", "")
        output_text = s.get("output", "")
        if input_text and output_text:
            pairs.append((input_text, output_text))
            
    return pairs

def generate_dataset():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tables_path = os.path.join(base_dir, "..", "all_tables.json")
    samples_path = os.path.join(base_dir, "..", "all_samples.json")
    out_dir = os.path.join(base_dir, "..", "data")
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading schema from {tables_path}...")
    schema_elements = load_schema(tables_path)
    print(f"Loaded {len(schema_elements)} included schema columns.")
    
    print(f"Loading samples from {samples_path}...")
    samples = load_samples(samples_path)
    print(f"Loaded {len(samples)} training pairs.")

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    
    def get_id(word):
        if word not in vocab:
            vocab[word] = len(vocab)
        return vocab[word]
        
    # Process schema: for now we represent each schema element by the average/first token 
    # to fit into the flat (schema_size) array format expected by the current C++ loader.
    # In a full model, this would be a 2D tensor (schema_size x schema_seq_len).
    schema_tokens_list = [tokenize(desc) for desc in schema_elements]
    
    for tokens in schema_tokens_list:
        for t in tokens:
            get_id(t)
            
    tokenized_samples = []
    for inp, out in samples:
        inp_toks = tokenize(inp)
        out_toks = tokenize(out)
        for t in inp_toks + out_toks:
            get_id(t)
        tokenized_samples.append((inp_toks, out_toks))
        
    vocab_list = sorted(vocab.keys(), key=lambda w: vocab[w])
    vocab_size = len(vocab_list)
    
    print(f"Vocabulary size: {vocab_size}")
    
    with open(os.path.join(out_dir, "breakwalls_vocab.txt"), "w", encoding="utf-8") as f:
        for w in vocab_list:
            f.write(f"{w}\n")

    # Serialize
    n_train = int(len(tokenized_samples) * 0.8)
    n_val = len(tokenized_samples) - n_train
    seq_len = 64
    schema_size = len(schema_elements)
    
    train_samples = tokenized_samples[:n_train]
    val_samples = tokenized_samples[n_train:]
    
    def pad_sequence(toks, length):
        ids = [vocab.get(t, vocab["[UNK]"]) for t in toks]
        if len(ids) > length:
            return ids[:length]
        return ids + [vocab["[PAD]"]] * (length - len(ids))

    def write_set(f, dataset, n):
        X = array.array('f', [0.0] * (n * seq_len))
        Y = array.array('f', [0.0] * (n * seq_len))
        
        # Schema is static for all queries, so we just repeat the schema token IDs
        # For simplicity, we just take the first token ID of the schema description to represent it.
        Schema = array.array('f', [0.0] * (n * schema_size))
        schema_ids = [vocab.get(toks[0], vocab["[UNK]"]) if toks else vocab["[UNK]"] for toks in schema_tokens_list]
        
        for i, (inp_toks, out_toks) in enumerate(dataset):
            inp_ids = pad_sequence(inp_toks, seq_len)
            out_ids = pad_sequence(out_toks, seq_len)
            
            for j in range(seq_len):
                X[i * seq_len + j] = float(inp_ids[j])
                Y[i * seq_len + j] = float(out_ids[j])
                
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
        
    print(f"Successfully generated dataset {out_bin}!")
    print(f"Train samples: {n_train}, Val samples: {n_val}, Seq Len: {seq_len}, Schema Size: {schema_size}")

if __name__ == "__main__":
    generate_dataset()
