"""
make_finetune_data.py -- Phase C: convert the existing Q->SQL corpus into an
instruction-tuning dataset for fine-tuning a LOCAL open coder LLM.

Reuses everything already built:
  - data/synthetic_dataset.json  : the question->SQL pairs (the fine-tune signal)
  - all_tables.json              : schema (tables + INCLUDE_IN_MODEL columns)
  - jargon_fusion.json           : business jargon -> column/fragment mappings

Output (chat/messages JSONL, consumed by trl / axolotl / llama-factory / unsloth):
  data/ft_train.jsonl, data/ft_val.jsonl
Each line: {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}

The SCHEMA + JARGON go in a FIXED system prompt (conditioning, not retrieval):
it's the same constant string every example, so the model is grounded on the
schema and tolerates minor schema edits without a full retrain -- but there is
no retrieval step at inference. Train/val split matches the generator's
held-out boundary so eval still measures unseen-phrasing generalization.
"""
import json
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def build_system_prompt(full=False):
    # COMPACT prompt (default): table names + jargon only. The full 325-column
    # enumeration made every sequence ~2.4k tokens -> OOM + ~5x slower training
    # on a 16GB T4. A fine-tuned model learns the columns from the 10k examples
    # (every column appears in many SQL outputs) and the guardrail validates
    # them at inference, so the in-context column list isn't needed for a
    # fine-tune. Pass full=True to restore the column lists (needs a bigger GPU).
    tables = json.load(open(os.path.join(BASE, "all_tables.json"), encoding="utf-8"))
    by_table = {}
    for it in tables:
        if it.get("INCLUDE_IN_MODEL") and it.get("COLUMN_NAME"):
            by_table.setdefault(it["TABLE_NAME"], []).append(it["COLUMN_NAME"])
    if full:
        schema = "SCHEMA (tables and their columns):\n" + "\n".join(
            f"  {t} ({', '.join(cols)})" for t, cols in by_table.items())
    else:
        schema = "TABLES: " + ", ".join(by_table.keys())

    jargon = json.load(open(os.path.join(BASE, "jargon_fusion.json"), encoding="utf-8"))
    jline = [f"  {e['keys'][0]} = {e['expansion']}" for e in jargon]

    return (
        "You are a T-SQL generator for an enterprise logistics database. "
        "Translate the user's question into a single valid T-SQL query. "
        "Output ONLY the SQL, no explanation.\n\n"
        + schema + "\n\n"
        "BUSINESS JARGON (term -> SQL meaning; use these exact columns/fragments):\n"
        + "\n".join(jline)
    )


def main():
    sys_prompt = build_system_prompt()
    data = json.load(open(os.path.join(BASE, "data", "synthetic_dataset.json"), encoding="utf-8"))

    # Same sequential 80/20 boundary the generator uses: train block then the
    # held-out (unseen-phrasing) val block. Keeps eval honest.
    n_train = int(len(data) * 0.8)

    def to_msg(x):
        return {"messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": x["input"]},
            {"role": "assistant", "content": x["output"]},
        ]}

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_msg(r), ensure_ascii=False) + "\n")

    out_dir = os.path.join(BASE, "data")
    dump(os.path.join(out_dir, "ft_train.jsonl"), data[:n_train])
    dump(os.path.join(out_dir, "ft_val.jsonl"), data[n_train:])

    print(f"system prompt: {len(sys_prompt)} chars (~{len(sys_prompt)//4} tokens)")
    print(f"ft_train.jsonl: {n_train} examples")
    print(f"ft_val.jsonl:   {len(data) - n_train} examples (held-out phrasings)")
    print("\n--- sample system prompt (first 600 chars) ---")
    print(sys_prompt[:600])
    print("\n--- sample example ---")
    ex = json.loads(json.dumps(to_msg(data[0])))
    print("USER:", ex["messages"][1]["content"][:80])
    print("SQL :", ex["messages"][2]["content"].replace("\n", " ")[:100])


if __name__ == "__main__":
    main()
