import json
import random
import os
import datetime

# Seed for reproducibility
random.seed(42)

# File paths
base_dir = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(base_dir, "..", "all_tables.json")
old_dataset_path = os.path.join(base_dir, "..", "data", "synthetic_dataset.json")
jargon_path = os.path.join(base_dir, "..", "jargon_dict.json")
output_path = old_dataset_path  # Overwrite existing dataset

# 1. Load data
with open(tables_path, 'r', encoding='utf-8') as f:
    schema_data = json.load(f)

# Extract tables and their valid columns
tables = {}
for item in schema_data:
    t_name = item.get("TABLE_NAME", "")
    c_name = item.get("COLUMN_NAME", "")
    if t_name and c_name:
        if t_name not in tables:
            tables[t_name] = []
        tables[t_name].append(c_name)

# Exclude some tables if they are not useful, or just use all
table_names = list(tables.keys())

with open(old_dataset_path, 'r', encoding='utf-8') as f:
    old_dataset = json.load(f)

# 2. Process Business Knowledge (30%)
business_samples = []
seen_pairs = set()

for sample in old_dataset:
    inp = sample.get("input", "").strip()
    out = sample.get("output", "").strip()
    pair = (inp, out)
    if pair not in seen_pairs:
        seen_pairs.add(pair)
        business_samples.append({"input": inp, "output": out})

# Augment business samples by paraphrasing inputs
paraphrases = [
    lambda x: f"Show me {x.lower()}" if not x.lower().startswith("show") else x,
    lambda x: f"Get {x.lower()}" if not x.lower().startswith("get") else x,
    lambda x: x.replace("What is", "Find").replace("What are", "List"),
    lambda x: x.replace("List", "Give me all").replace("Show", "Display"),
    lambda x: f"Can you tell me {x.lower()}?",
]

augmented_business = []
for sample in business_samples:
    augmented_business.append(sample)
    # Add 1 paraphrase
    new_inp = random.choice(paraphrases)(sample["input"])
    if new_inp != sample["input"]:
        augmented_business.append({"input": new_inp, "output": sample["output"]})

# Limit or expand to ~4500 samples
if len(augmented_business) > 4500:
    augmented_business = random.sample(augmented_business, 4500)
else:
    # If not enough, just duplicate some with more paraphrases
    while len(augmented_business) < 4500:
        sample = random.choice(business_samples)
        new_inp = random.choice(paraphrases)(sample["input"])
        augmented_business.append({"input": new_inp, "output": sample["output"]})

# 3. Generate General T-SQL (70% -> ~10500 samples)
general_samples = []

# Helper lists for random generation
question_prefixes = ["Show me", "List", "Get", "What is", "What are", "Find", "Display", "Give me", "Fetch", "Return", "Show", "Tell me"]
typo_prefixes = ["Hw many", "Wht is", "Lst", "Chekc", "nubmer", "fr", "te", "tp", "Sho me", "Gt"]
ops = ["=", ">", "<", ">=", "<=", "!="]
aggs = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
order_dirs = ["ASC", "DESC"]

values = {
    "string": ["'CUST001'", "'CUST002'", "'ACC-123'", "'MUM-2024'", "'OBD-1234'", "'PL-5678'", "'Mumbai'", "'Chennai'", "'Delhi'", "'Bangalore'", "'Hyderabad'", "'GATI'", "'DELHIVERY'", "'SAFEXPRESS'", "'Deco'", "'Marine'", "'Coatings'", "'Pending Picking'", "'Delivered'", "'Cancelled'"],
    "number": ["10", "100", "50", "0", "1", "5", "1000", "500"],
    "date": ["'2024-01-01'", "'2025-04-01'", "'2023-12-31'", "'2026-06-15'", "GETDATE()"]
}

def get_random_value():
    vt = random.choice(["string", "number", "date"])
    val = random.choice(values[vt])
    return val, val.replace("'", "") # val for SQL, unquoted for English

def get_table_col():
    t = random.choice(table_names)
    c = random.choice(tables[t])
    return t, c

def generate_prefix():
    if random.random() < 0.2:
        return random.choice(typo_prefixes)
    return random.choice(question_prefixes)

# P1: Simple SELECT with WHERE
for _ in range(1500):
    t, c = get_table_col()
    val_sql, val_eng = get_random_value()
    op = random.choice(ops)
    prefix = generate_prefix()
    
    inp = f"{prefix} {t} where {c} {op} {val_eng}"
    if random.random() < 0.5:
        c_sel = random.choice(tables[t])
        out = f"SELECT {c_sel} FROM {t} WHERE {c} {op} {val_sql};"
        inp = f"{prefix} {c_sel} from {t} where {c} is {val_eng}"
    else:
        out = f"SELECT * FROM {t} WHERE {c} {op} {val_sql};"
    
    general_samples.append({"input": inp, "output": out})

# P2: SELECT TOP N with ORDER BY
for _ in range(1500):
    t, c = get_table_col()
    n = random.choice([5, 10, 20, 50, 100])
    order = random.choice(order_dirs)
    prefix = generate_prefix()
    
    c_sel = random.choice(tables[t])
    inp = f"{prefix} top {n} {c_sel} from {t} ordered by {c} {order}"
    out = f"SELECT TOP {n} {c_sel} FROM {t} ORDER BY {c} {order};"
    general_samples.append({"input": inp, "output": out})

# P3: SELECT with GROUP BY + aggregate
for _ in range(1500):
    t, c = get_table_col()
    agg = random.choice(aggs)
    c_agg = random.choice(tables[t])
    prefix = generate_prefix()
    
    inp = f"{prefix} {agg} of {c_agg} grouped by {c} in {t}"
    out = f"SELECT {c}, {agg}({c_agg}) FROM {t} GROUP BY {c};"
    general_samples.append({"input": inp, "output": out})

# P4: SELECT with WHERE BETWEEN dates
for _ in range(1000):
    t, c = get_table_col()
    c_sel = random.choice(tables[t])
    d1_sql, d1_eng = random.choice(values["date"]), ""
    d2_sql, d2_eng = random.choice(values["date"]), ""
    while d1_sql == d2_sql or d1_sql == "GETDATE()" or d2_sql == "GETDATE()":
        d1_sql, _ = random.choice(values["date"])
        d2_sql, _ = random.choice(values["date"])
    d1_eng, d2_eng = d1_sql.replace("'", ""), d2_sql.replace("'", "")
    
    prefix = generate_prefix()
    inp = f"{prefix} {c_sel} from {t} where {c} is between {d1_eng} and {d2_eng}"
    out = f"SELECT {c_sel} FROM {t} WHERE {c} BETWEEN {d1_sql} AND {d2_sql};"
    general_samples.append({"input": inp, "output": out})

# P5: SELECT with JOIN (AN_CUSTOMER_VS_ASM_RSM and AN_LOGISTICS_TRACKER)
for _ in range(1000):
    t1 = "AN_CUSTOMER_VS_ASM_RSM"
    t2 = "AN_LOGISTICS_TRACKER"
    c1 = random.choice(tables[t1])
    c2 = random.choice(tables[t2])
    val_sql, val_eng = get_random_value()
    prefix = generate_prefix()
    
    inp = f"{prefix} {c1} and {c2} for {t1} and {t2} where {c2} is {val_eng}"
    out = f"SELECT A.{c1}, B.{c2} FROM {t1} A JOIN {t2} B ON A.CustomerCode = B.SoldToCustomerId WHERE B.{c2} = {val_sql};"
    general_samples.append({"input": inp, "output": out})

# P6: Multiple WHERE conditions (AND/OR)
for _ in range(1500):
    t, c1 = get_table_col()
    c2 = random.choice(tables[t])
    c_sel = random.choice(tables[t])
    v1_sql, v1_eng = get_random_value()
    v2_sql, v2_eng = get_random_value()
    op_logic = random.choice(["AND", "OR"])
    prefix = generate_prefix()
    
    inp = f"{prefix} {c_sel} from {t} where {c1} is {v1_eng} {op_logic.lower()} {c2} is {v2_eng}"
    out = f"SELECT {c_sel} FROM {t} WHERE {c1} = {v1_sql} {op_logic} {c2} = {v2_sql};"
    general_samples.append({"input": inp, "output": out})

# P7: ISNULL / COALESCE
for _ in range(1000):
    t, c = get_table_col()
    v_sql, v_eng = get_random_value()
    prefix = generate_prefix()
    
    inp = f"{prefix} {c} from {t}, if null use {v_eng}"
    out = f"SELECT ISNULL({c}, {v_sql}) FROM {t};"
    general_samples.append({"input": inp, "output": out})

# P8: Date functions
for _ in range(1000):
    t, c1 = get_table_col()
    c2 = random.choice(tables[t])
    c_sel = random.choice(tables[t])
    prefix = generate_prefix()
    
    inp = f"{prefix} difference in days between {c1} and {c2} for {t}"
    out = f"SELECT {c_sel}, DATEDIFF(day, {c1}, {c2}) FROM {t};"
    general_samples.append({"input": inp, "output": out})

# Combine and shuffle
final_dataset = business_samples + general_samples
random.shuffle(final_dataset)

# Save
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, indent=2)

print("Dataset generation complete!")
print(f"Total samples: {len(final_dataset)}")
print(f"Business samples: {len(business_samples)}")
print(f"General T-SQL samples: {len(general_samples)}")
