import os
import json
import struct
import array
from tokenizers import Tokenizer

def load_schema(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    schema_elements = []
    seen_tables = set()
    for item in tables:
        if item.get("INCLUDE_IN_MODEL", False):
            col_name = item.get("COLUMN_NAME", "")
            table_name = item.get("TABLE_NAME", "")
            synonyms = item.get("ALT_SYNONYMS", "")
            
            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                schema_elements.append((f"TABLE {table_name}", table_name.lower()))
                
            if col_name:
                desc = f"COLUMN {col_name} IN {table_name} {synonyms if synonyms else ''}"
                schema_elements.append((desc, col_name.lower()))
            
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
        
    schema_tokens_list = [tokenizer.encode(desc).ids for desc, _ in schema_elements]
    
    n_train = int(len(tokenized_samples) * 0.8)
    n_val = len(tokenized_samples) - n_train
    seq_len = 128 # Capped to reduce T^2 attention cost
    schema_size = len(schema_elements)
    max_schema_toks = 6 # Sub-tokens kept per schema element (full element, not just first BPE piece)
    
    train_samples = tokenized_samples[:n_train]
    val_samples = tokenized_samples[n_train:]
    
    pad_id = tokenizer.token_to_id("[pad]")
    if pad_id is None: pad_id = 0
    unk_id = tokenizer.token_to_id("[unk]")
    if unk_id is None: unk_id = 1
    
    # We need a SEP token to divide English prompt from SQL target
    sep_id = tokenizer.token_to_id("\n")
    if sep_id is None: sep_id = 1

    # EOS marks the end of the SQL target so the model learns to stop.
    eos_id = tokenizer.token_to_id("[EOS]")
    if eos_id is None: eos_id = unk_id

    # Full sub-token sequence per schema element, padded/truncated to max_schema_toks.
    schema_tok_matrix = []           # [schema_size, max_schema_toks]
    schema_vocab_ids = []            # [schema_size] copy target = explicit token id
    for i, toks in enumerate(schema_tokens_list):
        _, target_word = schema_elements[i]
        target_id = tokenizer.token_to_id(target_word)
        if target_id is None:
            print(f"Warning: target word '{target_word}' not found in vocab! Falling back to unk_id.")
            target_id = unk_id
            
        schema_vocab_ids.append(target_id)
        
        ids = toks[:max_schema_toks] if toks else [unk_id]
        ids = ids + [pad_id] * (max_schema_toks - len(ids))
        schema_tok_matrix.append(ids)

    def pad_sequence(ids, length):
        if len(ids) > length:
            return ids[:length]
        return ids + [pad_id] * (length - len(ids))

    schema_flat = [float(schema_tok_matrix[j][s]) for j in range(schema_size) for s in range(max_schema_toks)]

    def write_set(f, dataset, n):
        X = array.array('f', [0.0] * (n * seq_len))
        Y = array.array('f', [0.0] * (n * seq_len))
        # Full sub-token sequences per element: [n, schema_size * max_schema_toks]
        Schema = array.array('f', [0.0] * (n * schema_size * max_schema_toks))

        for i, (inp_ids, out_ids) in enumerate(dataset):
            combined = inp_ids + [sep_id] + out_ids + [eos_id]
            x_seq = combined[:-1]

            # Targets: ignore the prompt (and the sep position), supervise out_ids then EOS.
            y_labels = [-100] * len(inp_ids) + out_ids + [eos_id]

            x_pad = pad_sequence(x_seq, seq_len)
            if len(y_labels) > seq_len:
                y_pad = y_labels[:seq_len]
            else:
                y_pad = y_labels + [-100] * (seq_len - len(y_labels))

            for j in range(seq_len):
                X[i * seq_len + j] = float(x_pad[j])
                Y[i * seq_len + j] = float(y_pad[j])

            # Schema is global (identical every row); copy the flattened sub-token matrix.
            base = i * schema_size * max_schema_toks
            for k in range(schema_size * max_schema_toks):
                Schema[base + k] = schema_flat[k]

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
        f.write(struct.pack("i", max_schema_toks))

        # Copy-head target ids: one vocab id per schema element (first sub-token).
        array.array('f', [float(v) for v in schema_vocab_ids]).tofile(f)

        write_set(f, train_samples, n_train)
        write_set(f, val_samples, n_val)
        
    with open(os.path.join(out_dir, "schema_strings.txt"), "w", encoding="utf-8") as f:
        for desc in schema_elements:
            f.write(desc.lower().replace('\n', ' ') + '\n')
            
    with open(os.path.join(out_dir, "jargon_dict.txt"), "w", encoding="utf-8") as f:
        for k, v in jargon_dict.items():
            f.write(f"{k}|{v}\n")
        
    print(f"Successfully generated {out_bin} from synthetic dataset!")
    print(f"Train samples: {n_train}, Val samples: {n_val}, Seq Len: {seq_len}, Schema Size: {schema_size}, Max Schema Toks: {max_schema_toks}")

if __name__ == "__main__":
    generate_dataset()
