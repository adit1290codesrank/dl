import re
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
tables_path = os.path.join(base_dir, "all_tables.json")
samples_path = os.path.join(base_dir, "data", "synthetic_dataset.json")

def load_schema(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    schema_elements = []
    seen_tables = set()
    for item in tables:
        if item.get("INCLUDE_IN_MODEL", False):
            col_name = item.get("COLUMN_NAME", "")
            table_name = item.get("TABLE_NAME", "")
            
            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                schema_elements.append((f"TABLE {table_name}", table_name.lower()))
                
            if col_name:
                schema_elements.append((f"COLUMN {col_name} IN {table_name}", col_name.lower()))
            
    return schema_elements

schema_elements = load_schema(tables_path)

# Sort schema elements by length descending to match longest first
schema_dict = {}
for i, (desc, col) in enumerate(schema_elements):
    schema_dict[col] = i
sorted_cols = sorted(schema_dict.keys(), key=len, reverse=True)
pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_cols)) + r')\b', re.IGNORECASE)

with open(samples_path, 'r', encoding='utf-8') as f:
    samples = json.load(f)

for s in samples[:10]:
    out_sql = s["output"]
    print("Original:", out_sql)
    replaced = pattern.sub(lambda m: f"[SCHEMA_{schema_dict[m.group(1).lower()]}]", out_sql)
    print("Replaced:", replaced)
    print("-" * 50)
