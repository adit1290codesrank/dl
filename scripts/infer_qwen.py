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


def load_valid_identifiers():
    """Everything legitimately allowed in generated SQL (lowercased):
    schema tables/columns + procedure/fragment names & override columns mined
    from jargon_fusion.json (so EXEC procs and dbo.* fragments aren't flagged)."""
    tables = json.load(open(os.path.join(BASE, "all_tables.json"), encoding="utf-8"))
    valid = set(SQL_KEYWORDS)
    for it in tables:
        if it.get("INCLUDE_IN_MODEL"):
            if it.get("TABLE_NAME"): valid.add(it["TABLE_NAME"].lower())
            if it.get("COLUMN_NAME"): valid.add(it["COLUMN_NAME"].lower())
    jp = os.path.join(BASE, "jargon_fusion.json")
    if os.path.exists(jp):
        for e in json.load(open(jp, encoding="utf-8")):
            for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", e["expansion"]):
                valid.add(w.lower())     # ANDashBoardVehicleUtilization, dbo, HoursDifferenceF, override cols, ...
    return valid


# T-SQL keywords / functions that are NOT schema identifiers (don't flag them).
SQL_KEYWORDS = set("""select from where group by order asc desc top distinct count sum avg
min max as and or not in between is null like join on inner left right outer exec
isnull dateadd datediff getdate day month year having union all over partition
case when then else end null natureoftransaction dbo""".split())


def validate(sql, valid):
    """Flag identifiers that look like schema names but aren't valid -- while
    NOT flagging: @parameters, string literals, table aliases, or column
    ALIASES (an identifier in alias position: right after another identifier,
    a ')', or AS). This kills the false positives on procs/params/aliases while
    still catching a genuinely invented column."""
    body = re.sub(r"'[^']*'", "''", sql)        # blank string literals
    body = re.sub(r"@[A-Za-z_0-9]+", " ", body)  # drop @parameter names
    # tokenize into identifiers and single-char symbols, in order, to see context
    toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[(),.*=<>!]", body)
    unknown, prev = [], ""
    for t in toks:
        if re.match(r"[A-Za-z_]", t):            # it's an identifier
            tl = t.lower()
            # alias slot = right after a VALUE expression: a non-keyword
            # identifier (e.g. "VolumeInvoiced TotalVol"), a ")" (e.g.
            # "SUM(..) TotalVol"), or AS. NOT after a keyword like SELECT.
            prev_is_value_ident = bool(re.match(r"[A-Za-z_]", prev)) and prev not in SQL_KEYWORDS
            alias_pos = prev_is_value_ident or prev in (")", "as")
            if tl in valid or len(tl) <= 2 or alias_pos:
                pass                              # valid / table-alias / column-alias
            else:
                unknown.append(t)
        prev = t.lower()
    return sorted(set(unknown))


def tidy(sql):
    sql = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", sql)   # strip spaces inside quotes
    sql = sql.strip().strip("`").strip()
    # if the model wrapped it in a ```sql ... ``` fence, unwrap
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.S | re.I)
    if m: sql = m.group(1).strip()
    return sql


# Benchmark battery — each tier tests a different capability. Run with --bench.
BENCH = {
    "A. in-distribution (canonical phrasings — should be perfect)": [
        "What is the number of sales for MPY",
        "Who is the ASM for customer code ABC123",
        "Who are the RSM for Marine",
        "Give all the OBDs invoiced in April 2025",
        "Count the OBDs for the Marine business unit",
        "What is the Sales Order Number for OBD ABC123",
        "Vehicle Utilization for the period between 2025-01-01 to 2025-01-31",
        "Vendor or Transporter Dashboard for Safe Express for Jan 2025",
        "What is the DOT for GATI for Feb 2025",
    ],
    "B. paraphrases (unseen wordings — phrasing generalization)": [
        "roughly how many MPY sales did we book",
        "who's the area sales manager looking after account XYZ789",
        "which regional sales managers handle the marine unit",
        "show me every OBD that got billed in april 2025",
        "how many obds does powder have",
        "what's the sales order tied to obd ABC123",
    ],
    "C. patched gaps / harder business": [
        "which business unit moved the most volume",
        "vehicle utilization for transporter ABC123 during March 2025",
        "vehicle utilization by transporter and vehicle type for May 2025",
        "list top warehouses where OBDs are pending dispatch over 7 days",
        "who is the RSM for OBD number ABC123",
        "set transporter as Safexpress for the following OBDs\nOBD123\nOBD456\nOBD678",
    ],
    "D. novel / out-of-family (compositional — LLM bonus)": [
        "average volume invoiced per warehouse for obds dispatched in 2025",
        "list customers with more than 5 cancelled obds",
        "which warehouse has the most pending dispatch obds",
        "total volume invoiced for marine and protective coating only",
        "count distinct transporters per warehouse",
    ],
    "E. robustness (typos / terse / out-of-domain)": [
        "wat is the asm for cust ABC123",
        "hw many mpy sales",
        "obds in cochin warehouse pending dispatch",
        "what is the weather today",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=str, default=None)
    ap.add_argument("--bench", action="store_true", help="run the full test battery")
    args = ap.parse_args()

    sys_prompt = build_system_prompt()
    valid = load_valid_identifiers()

    tok = AutoTokenizer.from_pretrained(ADAPTER)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    def gen_sql(q):
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        sql = tidy(gen)
        return sql, validate(sql, valid)

    def answer(q):
        sql, bad = gen_sql(q)
        print("SQL  :", sql)
        print("WARN : not in schema:" , bad) if bad else print("OK   : all identifiers valid")

    if args.bench:
        import io
        out_path = os.path.join(BASE, "bench_results.txt")
        lines, n_ok = [], 0
        total = sum(len(v) for v in BENCH.values())
        for tier, qs in BENCH.items():
            lines.append("\n" + "=" * 70 + "\n" + tier + "\n" + "=" * 70)
            for q in qs:
                sql, bad = gen_sql(q)
                ok = not bad
                n_ok += ok
                lines.append(f"\nQ   : {q}")
                lines.append(f"SQL : {sql}")
                lines.append("WARN: not in schema: " + str(bad) if bad else "OK  : valid")
        report = "\n".join(lines)
        report += f"\n\n{'=' * 70}\nGuardrail-clean: {n_ok}/{total}\n"
        print(report)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n(written to {out_path} -- paste it back for review)")
        return

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
