"""
value_canon.py -- deterministic VALUE canonicalization (entity/value linking).

The model picks the right column; this snaps the literal it compared against to
the EXACT value the DB stores, so "safe express" / "blr" / "banaglore" match
'SAFE EXPRESS' / 'Bangalore'. Same "deterministic where it must be exact"
principle as schema-name copying -- applied to values.

Only CLOSED-SET categorical columns are touched. Open values (ids, dates,
customer codes) are left exactly as written -- never fuzzy-match those.

PRODUCTION: replace these lists with `SELECT DISTINCT <col>` from the live DB
(one-time export per categorical column), and grow ALIASES with the business
abbreviations your users actually type.
"""
import re
import difflib

# Canonical value sets per categorical column (exact DB spellings/casing).
CATEGORICAL = {
    "SMU": ["Marine", "Protective Coating", "Deco", "Coatings", "Powder"],
    "SiteId": ["Bangalore", "Cochin", "Chennai", "Mumbai", "Delhi",
               "Hyderabad", "Kolkata", "Pune", "Indore", "Nagpur"],
    "Transporter": ["SAFE EXPRESS", "CUBE LOGISTICS", "SYNERGY BAXIS", "GATI",
                    "DELHIVERY", "TCI EXPRESS", "VRL LOGISTICS", "BLUE DART"],
    "NatureOfTransaction": ["Sales", "Stock Transfer"],
    "PendingStatus": ["Pending Dispatch", "Pending Picking", "Cancelled", "Delivered"],
    "Business": ["Deco", "Coatings", "Marine", "Powder"],
}

# Business abbreviations / nicknames -> canonical value (lowercased keys).
ALIASES = {
    "SiteId": {
        "blr": "Bangalore", "bengaluru": "Bangalore", "bangalore": "Bangalore",
        "bom": "Mumbai", "mum": "Mumbai", "del": "Delhi", "ncr": "Delhi",
        "hyd": "Hyderabad", "kol": "Kolkata", "ccu": "Kolkata", "cok": "Cochin",
        "kochi": "Cochin", "maa": "Chennai", "madras": "Chennai",
    },
    "Transporter": {
        "safex": "SAFE EXPRESS", "safexpress": "SAFE EXPRESS",
        "safe express": "SAFE EXPRESS", "cube": "CUBE LOGISTICS",
        "synergy": "SYNERGY BAXIS", "vrl": "VRL LOGISTICS", "tci": "TCI EXPRESS",
        "bluedart": "BLUE DART", "blue dart": "BLUE DART",
    },
}

FUZZY_CUTOFF = 0.72


def canon_value(col, val):
    """Return (canonical_value, changed?). Leaves val untouched if col isn't
    categorical or no confident match is found."""
    cset = CATEGORICAL.get(col)
    if cset is None:
        return val, False                      # open column -> never touch
    low = val.strip().lower()
    # 1. exact, case-insensitive
    for c in cset:
        if c.lower() == low:
            return c, (c != val)
    # 2. alias map
    amap = ALIASES.get(col, {})
    if low in amap:
        return amap[low], True
    # 3. fuzzy (compare lowercased, return canonical casing)
    lows = [c.lower() for c in cset]
    m = difflib.get_close_matches(low, lows, n=1, cutoff=FUZZY_CUTOFF)
    if m:
        return cset[lows.index(m[0])], True
    return val, False                          # no confident match -> leave as-is


def canonicalize_sql(sql):
    """Snap categorical literals in `col = 'v'` and `col IN ('v1','v2')` to
    their canonical DB values. Column may be qualified (B.SiteId) -- we match
    on the bare column name."""
    def fix_eq(m):
        col, val = m.group(1), m.group(2)
        return f"{col} = '{canon_value(col, val)[0]}'"
    sql = re.sub(r"(\w+)\s*=\s*'([^']*)'", fix_eq, sql)

    def fix_in(m):
        col, body = m.group(1), m.group(2)
        vals = re.findall(r"'([^']*)'", body)
        joined = ", ".join(f"'{canon_value(col, v)[0]}'" for v in vals)
        return f"{col} IN ({joined})"
    sql = re.sub(r"(\w+)\s+IN\s*\(([^)]*)\)", fix_in, sql, flags=re.I)

    # CB_OBDMassUpdate passes (field, value) as positional string args:
    #   EXEC CB_OBDMassUpdate '_user_', '<ids>', '<Field>', '<Value>'
    # If <Field> names a categorical column, canonicalize <Value> against it.
    # (A non-quoted last arg like `, 1` for Cancelled won't match -> left alone.)
    def fix_massupdate(m):
        head, field, mid, value, tail = m.groups()
        return head + field + mid + canon_value(field, value)[0] + tail
    sql = re.sub(
        r"(CB_OBDMassUpdate\s*'[^']*'\s*,\s*'[^']*'\s*,\s*')([^']*)('\s*,\s*')([^']*)(')",
        fix_massupdate, sql, flags=re.I)
    return sql


if __name__ == "__main__":
    tests = [
        "SELECT * FROM AN_LOGISTICS_TRACKER WHERE Transporter = 'safe express'",
        "WHERE SiteId = 'blr'",
        "WHERE SiteId = 'banaglore'",                 # typo -> fuzzy
        "WHERE SMU IN ( 'marine', 'protective coating')",
        "WHERE NatureOfTransaction = 'sales'",
        "WHERE CustomerCode = 'ABC123'",              # open -> unchanged
        "WHERE PickListId = 'OBD123'",                # open -> unchanged
        "WHERE B.SiteId = 'bengaluru'",               # qualified
        "EXEC CB_OBDMassUpdate '_CB_USERNAME_', 'OBD123', 'Transporter', 'safexpress'",
    ]
    for t in tests:
        print(t, "\n  ->", canonicalize_sql(t), "\n")
