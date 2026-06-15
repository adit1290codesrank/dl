"""
infer_qwen.py -- Phase C inference: fine-tuned Qwen2.5-Coder-7B + the GUARDRAIL
layer (the part of the from-scratch project that carries over and matters most).

The LLM supplies boundless language; the guardrail supplies trust:
  1. generate SQL from the question (fine-tuned, schema baked in via system prompt),
  2. VALIDATE every table/column the SQL references against the real schema,
     flagging anything hallucinated -- a raw LLM's #1 failure for enterprise SQL,
  3. light formatting tidy.
This is the deterministic-emission idea reframed: we can't force the token-level
copy on an LLM, but we CAN reject/flag any output that names a column the DB
doesn't have. Optional next step: grammar-constrained decoding for a hard guarantee.

Deps: pip install transformers peft bitsandbytes accelerate
Run:  python scripts/infer_qwen.py            # REPL
      python scripts/infer_qwen.py --q "..."  # single question

NOTE: untested locally (needs a GPU with the adapter trained). First run is the
smoke test.
"""
import os
import re
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MUST match the base the adapter was trained on (finetune_qwen.py) -- the LoRA
# shapes are tied to the base hidden size (14B=5120, 7B=3584).
MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
ADAPTER = os.path.join(BASE, "weights", "qwen_sql_lora")

# Reuse the exact system prompt the model was fine-tuned with.
sys.path.insert(0, os.path.join(BASE, "scripts"))
from make_finetune_data import build_system_prompt  # noqa: E402


def load_schema_identifiers():
    """All valid table + column names (lowercased) for the validation guardrail."""
    tables = json.load(open(os.path.join(BASE, "all_tables.json"), encoding="utf-8"))
    tabs, cols = set(), set()
    for it in tables:
        if it.get("INCLUDE_IN_MODEL"):
            if it.get("TABLE_NAME"): tabs.add(it["TABLE_NAME"].lower())
            if it.get("COLUMN_NAME"): cols.add(it["COLUMN_NAME"].lower())
    return tabs, cols


# T-SQL keywords / functions that are NOT schema identifiers (don't flag them).
SQL_KEYWORDS = set("""select from where group by order asc desc top distinct count sum avg
min max as and or not in between is null like join on inner left right outer exec
isnull dateadd datediff getdate day month year having union all count(*) over partition
case when then else end null natureoftransaction""".split())


def validate(sql, tabs, cols):
    """Flag identifiers that look like schema names but aren't in the schema."""
    # candidate identifiers: alnum/underscore tokens that aren't quoted literals,
    # numbers, or SQL keywords.
    body = re.sub(r"'[^']*'", "''", sql)              # blank out string literals
    toks = re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", body)
    valid = tabs | cols | SQL_KEYWORDS
    # also accept dotted aliases a.Col and the proc/fragment names the model emits
    unknown = []
    for t in toks:
        tl = t.lower()
        if tl in valid: continue
        if len(tl) <= 2: continue                     # table aliases a, b, ...
        # column referenced as alias.Column -> check the column part separately
        unknown.append(t)
    # keep only ones that really aren't anywhere in schema (dedupe, drop dups)
    return sorted(set(u for u in unknown if u.lower() not in valid))


def tidy(sql):
    sql = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", sql)   # strip spaces inside quotes
    sql = sql.strip().strip("`").strip()
    # if the model wrapped it in a ```sql ... ``` fence, unwrap
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.S | re.I)
    if m: sql = m.group(1).strip()
    return sql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=str, default=None)
    args = ap.parse_args()

    sys_prompt = build_system_prompt()
    tabs, cols = load_schema_identifiers()

    tok = AutoTokenizer.from_pretrained(ADAPTER)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    def answer(q):
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        sql = tidy(gen)
        bad = validate(sql, tabs, cols)
        print("SQL  :", sql)
        if bad:
            print("WARN : identifiers not in schema (possible hallucination):", bad)
        else:
            print("OK   : all identifiers valid against schema")

    if args.q:
        answer(args.q)
        return
    print("Fine-tuned Qwen text-to-SQL. Type a question, or 'exit'.")
    while True:
        q = input("\nUSER > ").strip()
        if not q or q == "exit": break
        answer(q)


if __name__ == "__main__":
    main()
