import json
import os

with open('all_tables.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tables = set()
columns = set()
synonyms = set()

for d in data:
    tables.add(d['TABLE_NAME'])
    columns.add(d['COLUMN_NAME'])
    if d['ALT_SYNONYMS']:
        for s in d['ALT_SYNONYMS'].split(','):
            synonyms.add(s.strip())

print(f"Total Tables: {len(tables)}")
print(f"Total Columns: {len(columns)}")
print(f"Total Distinct Jargon Terms (Synonyms): {len(synonyms)}")
print("Some Jargon Terms:", list(synonyms)[:15])
