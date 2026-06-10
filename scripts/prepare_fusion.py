"""
prepare_fusion.py -- dataset builder for the Deep Memory-Fusion decoder.

Differences vs prepare_breakwalls.py (kept as the reproducible baseline):
  1. COMPACT VOCAB: atomic IDs sit directly after the real BPE vocab instead of
     at 50000+. Layout (single source of truth, written into the bin header):
        [0, V_bpe)            real BPE tokens (pad=0, unk=1, cls=2, sep=3, eos=4)
        [V_bpe, V_bpe+S)      schema element atomic IDs (tables + columns)
        [V_bpe+S, V)          jargon-fragment atomic IDs (expansions that are
                              NOT plain column names, e.g. MPY's WHERE clause)
  2. JARGON MOVES INTO THE MEMORY BANK: inputs are NOT string-replaced anymore.
     The question keeps "MPY"/"ASM" literally; a memory row with the term as its
     key points at the column ID (ASM -> SalesPersonName) or at a fragment ID
     (MPY). Selection is learned similarity; emission is deterministic.
  3. MEMORY BANK with multiple key rows per emit ID: one row per schema element,
     one extra row per ALT_SYNONYMS phrase, one row per jargon term.
  4. Memory is stored ONCE in the bin (it is global), not per example.
"""
import os
import json
import struct
import array
import re
import sys
from tokenizers import Tokenizer

SEQ_LEN = 128
MAX_MEM_TOKS = 8


def load_schema(filepath):
    """Returns list of dicts: {kind, name, table, key_text, synonyms}."""
    with open(filepath, 'r', encoding='utf-8') as f:
        tables = json.load(f)

    elements = []
    seen_tables = set()
    for item in tables:
        if not item.get("INCLUDE_IN_MODEL", False):
            continue
        col_name = item.get("COLUMN_NAME", "") or ""
        table_name = item.get("TABLE_NAME", "") or ""
        synonyms_raw = item.get("ALT_SYNONYMS") or ""

        if table_name and table_name not in seen_tables:
            seen_tables.add(table_name)
            elements.append({
                "kind": "table", "name": table_name, "table": table_name,
                "key_text": f"TABLE {table_name}", "synonyms": [],
            })
        if col_name:
            syns = [s.strip() for s in synonyms_raw.split(",") if s.strip()]
            elements.append({
                "kind": "column", "name": col_name, "table": table_name,
                "key_text": f"COLUMN {col_name} IN {table_name}", "synonyms": syns,
            })
    return elements


def fragment_pattern(fragment):
    """Whitespace-flexible regex for a SQL fragment. The dataset renders the
    MPY expansion as "SMU IN( 'Protective..." with embedded newlines, so we
    match token-by-token with \\s* in between."""
    toks = re.findall(r"\w+|[^\w\s]", fragment)
    pat = r"\s*".join(map(re.escape, toks))
    if fragment[:1].isalnum() or fragment[:1] == "_":
        pat = r"(?<!\w)" + pat
    if fragment[-1:].isalnum() or fragment[-1:] == "_":
        pat = pat + r"(?!\w)"
    return re.compile(pat, re.IGNORECASE)


def generate_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base_dir, "..")
    out_dir = os.path.join(root, "data")

    tokenizer_path = os.path.join(out_dir, "bpe_tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("Error: Run train_bpe.py first!")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    V_bpe = tokenizer.get_vocab_size()
    print(f"BPE Vocab Size (V_bpe): {V_bpe}")

    pad_id = tokenizer.token_to_id("[pad]") or 0
    unk_id = tokenizer.token_to_id("[unk]") or 1
    sep_id = tokenizer.token_to_id("[sep]")
    eos_id = tokenizer.token_to_id("[eos]")
    assert sep_id is not None and eos_id is not None, "tokenizer missing [sep]/[eos]"

    schema_elements = load_schema(os.path.join(root, "all_tables.json"))
    S = len(schema_elements)
    print(f"Schema elements: {S}")

    # name (lowercase) -> schema element index; last occurrence wins, matching
    # the baseline's behavior for duplicate column names across tables.
    schema_dict = {}
    for i, el in enumerate(schema_elements):
        schema_dict[el["name"].lower()] = i

    # jargon_fusion.json: [{"keys": [term, ...], "expansion": str}, ...]
    # mined from the 45 hand-written seeds in all_samples.json plus the
    # original jargon_dict.json. Falls back to jargon_dict.json if absent.
    jargon_path = os.path.join(root, "jargon_fusion.json")
    if os.path.exists(jargon_path):
        with open(jargon_path, 'r', encoding='utf-8') as f:
            jargon_entries = json.load(f)
    else:
        with open(os.path.join(root, "jargon_dict.json"), 'r', encoding='utf-8') as f:
            jargon_entries = [{"keys": [k], "expansion": v}
                              for k, v in json.load(f).items()]

    # Classify jargon: expansion that is a known schema name becomes a set of
    # synonym rows (emit = that element's ID); anything else (SQL fragments,
    # procedure names, report columns) gets its own atomic fragment ID.
    syn_jargon = []     # (key, schema element idx)
    frag_entries = []   # {"keys": [...], "expansion": str}
    for e in jargon_entries:
        if e["expansion"].lower() in schema_dict:
            idx = schema_dict[e["expansion"].lower()]
            syn_jargon.extend((k, idx) for k in e["keys"])
        else:
            frag_entries.append(e)
    J = len(frag_entries)
    V = V_bpe + S + J
    print(f"Jargon: {len(syn_jargon)} synonym keys -> schema columns, "
          f"{J} fragment entries:")
    for e in frag_entries:
        print(f"  {e['keys'][0]!r:32s} -> {e['expansion'][:70]}")
    print(f"Total vocab V = {V_bpe} + {S} + {J} = {V}")

    def schema_vid(idx):
        return V_bpe + idx

    def frag_vid(j):
        return V_bpe + S + j

    # ---- Memory bank: (key_text, emit_id) -------------------------------
    memory_rows = []
    for i, el in enumerate(schema_elements):
        memory_rows.append((el["key_text"], schema_vid(i)))
    for i, el in enumerate(schema_elements):
        for syn in el["synonyms"]:
            memory_rows.append((syn, schema_vid(i)))
    for term, idx in syn_jargon:
        memory_rows.append((term, schema_vid(idx)))
    n_frag_keys = 0
    for j, e in enumerate(frag_entries):
        for k in e["keys"]:
            memory_rows.append((k, frag_vid(j)))
            n_frag_keys += 1
    M = len(memory_rows)
    print(f"Memory rows M = {M} ({S} schema + "
          f"{M - S - len(syn_jargon) - n_frag_keys} synonyms + "
          f"{len(syn_jargon)} jargon-column keys + {n_frag_keys} fragment keys)")

    mem_tokens = []
    for key_text, _ in memory_rows:
        ids = tokenizer.encode(key_text).ids[:MAX_MEM_TOKS] or [unk_id]
        mem_tokens.append(ids + [pad_id] * (MAX_MEM_TOKS - len(ids)))
    mem_emit_ids = [vid for _, vid in memory_rows]

    # ---- Expansion table: atomic id -> exact emission string ------------
    expansions = {}
    for i, el in enumerate(schema_elements):
        expansions[schema_vid(i)] = el["name"]
    for j, e in enumerate(frag_entries):
        expansions[frag_vid(j)] = e["expansion"]

    # ---- Target encoding --------------------------------------------------
    # Order matters twice:
    #   1. Fragments before schema names (the MPY fragment contains the column
    #      names SMU and NatureOfTransaction which the column regex would eat).
    #   2. Longer fragments before shorter ones (the HoursDifferenceF(...)
    #      fragments contain the ISNULL(...) dispatch-date fragments).
    frag_order = sorted(range(J), key=lambda j: -len(frag_entries[j]["expansion"]))
    frag_pats = [(frag_entries[j]["keys"][0],
                  fragment_pattern(frag_entries[j]["expansion"]),
                  frag_vid(j))
                 for j in frag_order]
    sorted_names = sorted(schema_dict.keys(), key=len, reverse=True)
    col_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_names)) + r')\b',
                             re.IGNORECASE)

    frag_counts = {label: 0 for (label, _, _) in frag_pats}

    def encode_target(out_str):
        segments = [out_str]
        for term, pat, vid in frag_pats:
            nxt = []
            for seg in segments:
                if isinstance(seg, int):
                    nxt.append(seg)
                    continue
                last = 0
                for m in pat.finditer(seg):
                    if m.start() > last:
                        nxt.append(seg[last:m.start()])
                    nxt.append(vid)
                    frag_counts[term] += 1
                    last = m.end()
                if last < len(seg):
                    nxt.append(seg[last:])
            segments = nxt
        out_ids = []
        for seg in segments:
            if isinstance(seg, int):
                out_ids.append(seg)
                continue
            last = 0
            for m in col_pattern.finditer(seg):
                if m.start() > last:
                    out_ids.extend(tokenizer.encode(seg[last:m.start()]).ids)
                out_ids.append(schema_vid(schema_dict[m.group(1).lower()]))
                last = m.end()
            if last < len(seg):
                out_ids.extend(tokenizer.encode(seg[last:]).ids)
        return out_ids

    def decode_ids(ids):
        parts, buf = [], []
        for i in ids:
            if i >= V_bpe:
                if buf:
                    parts.append(tokenizer.decode(buf))
                    buf = []
                parts.append(expansions[i])
            else:
                buf.append(i)
        if buf:
            parts.append(tokenizer.decode(buf))
        return " ".join(parts)

    def squash(s):
        return re.sub(r"\s+", "", s).lower()

    # ---- Load samples ------------------------------------------------------
    with open(os.path.join(out_dir, "synthetic_dataset.json"), 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if "--dedup" in sys.argv:
        seen = set()
        deduped = []
        for s in samples:
            key = (s["input"], s["output"])
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        print(f"DEDUP: {len(samples)} -> {len(deduped)} unique pairs")
        samples = deduped

    if "--mini" in sys.argv:
        print("MINI MODE: Slicing dataset to first 100 samples!")
        samples = samples[:100]

    print(f"Tokenizing {len(samples)} examples...")
    tokenized = []
    roundtrip_fail = 0
    for s in samples:
        # NO jargon replacement: the question keeps MPY/ASM/... literally.
        inp_ids = tokenizer.encode(s["input"]).ids
        out_ids = encode_target(s["output"])
        assert all(0 <= i < V for i in out_ids), f"target id out of range: {s['output']}"
        assert all(0 <= i < V_bpe for i in inp_ids), "input id outside BPE range"
        if squash(decode_ids(out_ids)) != squash(s["output"]):
            roundtrip_fail += 1
            if roundtrip_fail <= 3:
                print("  [roundtrip-fail]")
                print("    orig   :", repr(s["output"][:140]))
                print("    decoded:", repr(decode_ids(out_ids)[:140]))
        tokenized.append((inp_ids, out_ids))

    # ---- Hard assertions ---------------------------------------------------
    rt_ok = 1.0 - roundtrip_fail / max(1, len(tokenized))
    print(f"Round-trip decode equality: {rt_ok * 100:.2f}% "
          f"({roundtrip_fail} failures)")
    for label, count in frag_counts.items():
        print(f"Fragment replacements for {label!r}: {count}")
    if frag_counts.get("MPY", 1) == 0:
        if any("Protective Coating" in s["output"] for s in samples):
            raise AssertionError("MPY fragment matched 0 times but its expansion "
                                 "appears in outputs -- fragment regex is broken!")
    assert rt_ok >= 0.99, "Round-trip decode equality below 99% -- encoding is lossy!"

    # ---- Sequence assembly -------------------------------------------------
    n_train = int(len(tokenized) * 0.8)
    n_val = len(tokenized) - n_train
    train_samples = tokenized[:n_train]
    val_samples = tokenized[n_train:]

    truncated_supervised = 0

    def write_set(f, dataset, n):
        nonlocal truncated_supervised
        X = array.array('f', [0.0] * (n * SEQ_LEN))
        Y = array.array('f', [0.0] * (n * SEQ_LEN))
        for i, (inp_ids, out_ids) in enumerate(dataset):
            combined = inp_ids + [sep_id] + out_ids + [eos_id]
            x_seq = combined[:-1]
            y_labels = [-100] * len(inp_ids) + out_ids + [eos_id]
            if len(y_labels) > SEQ_LEN:
                truncated_supervised += len(y_labels) - SEQ_LEN
                y_labels = y_labels[:SEQ_LEN]
            x_seq = (x_seq + [pad_id] * SEQ_LEN)[:SEQ_LEN]
            y_labels = y_labels + [-100] * (SEQ_LEN - len(y_labels))
            for j in range(SEQ_LEN):
                X[i * SEQ_LEN + j] = float(x_seq[j])
                Y[i * SEQ_LEN + j] = float(y_labels[j])
        X.tofile(f)
        Y.tofile(f)

    out_bin = os.path.join(out_dir, "fusion.bin")
    with open(out_bin, "wb") as f:
        for v in (n_train, n_val, SEQ_LEN, V, V_bpe, M, MAX_MEM_TOKS, S, J):
            f.write(struct.pack("i", v))
        array.array('f', [float(v) for v in mem_emit_ids]).tofile(f)
        array.array('f', [float(t) for row in mem_tokens for t in row]).tofile(f)
        write_set(f, train_samples, n_train)
        write_set(f, val_samples, n_val)

    print(f"Truncated supervised tokens: {truncated_supervised}")

    with open(os.path.join(out_dir, "fusion_expansions.txt"), "w", encoding="utf-8") as f:
        for vid in sorted(expansions):
            f.write(f"{vid}|{expansions[vid].replace(chr(10), ' ')}\n")
    with open(os.path.join(out_dir, "fusion_memory.txt"), "w", encoding="utf-8") as f:
        for key_text, vid in memory_rows:
            f.write(f"{vid}|{key_text}\n")

    print(f"Successfully generated {out_bin}")
    print(f"Train: {n_train}, Val: {n_val}, SeqLen: {SEQ_LEN}, "
          f"V: {V}, V_bpe: {V_bpe}, M: {M}, S: {S}, J: {J}")


if __name__ == "__main__":
    generate_dataset()
