import json
import random
import re
import datetime
import os

# --- Configurations ---
PHASE_A_COUNT = 2000
PHASE_B_COPIES = 50
TYPO_PROBABILITY = 0.2

# --- Helpers ---
def random_date(start_year=2020, end_year=2026):
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return datetime.date(year, month, day).strftime('%Y-%m-%d')

def random_string(length=6):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))

def inject_typo(text):
    if random.random() > TYPO_PROBABILITY:
        return text
    words = text.split()
    if not words: return text
    idx = random.randint(0, len(words) - 1)
    word = list(words[idx])
    if len(word) > 4:
        # Swap two adjacent characters
        c_idx = random.randint(1, len(word) - 2)
        word[c_idx], word[c_idx+1] = word[c_idx+1], word[c_idx]
    elif len(word) > 2:
        # Drop a character
        c_idx = random.randint(1, len(word) - 2)
        word.pop(c_idx)
    words[idx] = "".join(word)
    return " ".join(words)

def generate_dataset():
    print("Loading schema and samples...")
    with open('all_tables.json', 'r') as f:
        schema = json.load(f)
    
    with open('all_samples.json', 'r') as f:
        samples = json.load(f)

    # Parse Schema
    tables = {}
    for row in schema:
        t_name = row['TABLE_NAME']
        c_name = row['COLUMN_NAME']
        if t_name not in tables:
            tables[t_name] = []
        tables[t_name].append(c_name)

    dataset = []
    
    # ==========================================
    # PHASE A: Foundational T-SQL Grammar
    # ==========================================
    print(f"Generating Phase A: {PHASE_A_COUNT} basic T-SQL examples...")
    
    table_names = list(tables.keys())
    
    for i in range(PHASE_A_COUNT):
        t = random.choice(table_names)
        cols = tables[t]
        c1 = random.choice(cols)
        c2 = random.choice(cols)
        c3 = random.choice(cols)
        val = random_string()
        
        q_type = random.randint(1, 5)
        
        if q_type == 1: # Basic Select
            sql = f"SELECT {c1} FROM {t} WHERE {c2} = '{val}';"
            eng = random.choice([
                f"What is the {c1} in {t} where {c2} is {val}?",
                f"Find {c1} from {t} for {c2} {val}",
                f"Get the {c1} when {c2} equals {val}"
            ])
        elif q_type == 2: # Aggregation
            sql = f"SELECT {c1}, SUM({c2}) FROM {t} GROUP BY {c1};"
            eng = random.choice([
                f"Show me total {c2} grouped by {c1}",
                f"What is the sum of {c2} for each {c1}?",
                f"List {c1} and their total {c2}"
            ])
        elif q_type == 3: # Sorting & Top
            sql = f"SELECT TOP 5 {c1} FROM {t} ORDER BY {c2} DESC;"
            eng = random.choice([
                f"Top 5 {c1} by {c2}",
                f"What are the highest 5 {c1} based on {c2}?",
                f"Show top 5 {c1} ordered by {c2} descending"
            ])
        elif q_type == 4: # Count
            sql = f"SELECT COUNT({c1}) FROM {t} WHERE {c2} > 10;"
            eng = random.choice([
                f"Count the number of {c1} where {c2} is greater than 10",
                f"How many {c1} have {c2} > 10?",
                f"Total count of {c1} filtered by {c2} over 10"
            ])
        else: # Date Filtering
            d1 = random_date()
            d2 = random_date()
            sql = f"SELECT {c1} FROM {t} WHERE {c2} BETWEEN '{d1}' AND '{d2}';"
            eng = random.choice([
                f"Get {c1} where {c2} is between {d1} and {d2}",
                f"Show {c1} from {d1} to {d2} based on {c2}",
                f"List {c1} filtering {c2} from {d1} to {d2}"
            ])
            
        dataset.append({"input": inject_typo(eng), "output": sql})

    # ==========================================
    # PHASE B: Domain Knowledge Extrapolation
    # ==========================================
    print(f"Generating Phase B: Extrapolating {len(samples)} examples {PHASE_B_COPIES} times...")
    
    locations = ['Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Cochin', 'Kolkata', 'Hyderabad', 'Pune']
    transporters = ['Safe Express', 'Delhivery', 'BlueDart', 'Gati', 'VRL', 'FedEx', 'DHL']
    plants = ['Q80D', 'Q8CD', 'P10A', 'M20B', 'X99Z']
    smu_list = ['Marine', 'Protective Coating', 'Deco', 'Industrial', 'Powder']
    
    for sample in samples:
        orig_eng = sample['input']
        orig_sql = sample['output']
        
        for _ in range(PHASE_B_COPIES):
            eng = orig_eng
            sql = orig_sql
            
            # Extract and Swap IDs
            if 'ABC123' in eng:
                new_id = random_string(6)
                eng = eng.replace('ABC123', new_id)
                sql = sql.replace('ABC123', new_id)
                
            # Extract and Swap Dates
            dates_in_sql = re.findall(r"'\d{4}-\d{2}-\d{2}'", sql)
            if len(dates_in_sql) == 2:
                d1, d2 = random_date(), random_date()
                if d1 > d2: d1, d2 = d2, d1
                sql = sql.replace(dates_in_sql[0], f"'{d1}'", 1)
                sql = sql.replace(dates_in_sql[1], f"'{d2}'", 1)
                # Naive english date swap (in real pipeline, we'd use regex on English too)
                eng = eng.replace('2025-01-01', d1).replace('2025-01-31', d2)
                eng = eng.replace('april', datetime.datetime.strptime(d1, "%Y-%m-%d").strftime("%B").lower())
                
            # Swap Locations
            if 'Bangalore' in eng:
                loc = random.choice(locations)
                eng = eng.replace('Bangalore', loc)
                sql = sql.replace('Bangalore', loc)
            if 'Cochin' in eng:
                loc = random.choice(locations)
                eng = eng.replace('Cochin', loc)
                sql = sql.replace('Cochin', loc)
                
            # Swap Transporters
            if 'Safe Express' in eng:
                tr = random.choice(transporters)
                eng = eng.replace('Safe Express', tr)
                sql = sql.replace('SAFE EXPRESS', tr.upper()).replace('Safe Express', tr)
                
            # Swap Plants
            for p in plants:
                if p in eng:
                    new_p = random.choice(plants)
                    eng = eng.replace(p, new_p)
                    sql = sql.replace(p, new_p)
                    
            dataset.append({"input": inject_typo(eng), "output": sql})
            
    # Shuffle and save
    random.shuffle(dataset)
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/synthetic_dataset.json'
    with open(out_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Success! Generated {len(dataset)} highly varied training examples at {out_path}.")

if __name__ == "__main__":
    generate_dataset()
