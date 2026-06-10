"""
value_slots.py -- deterministic literal-value delexicalization, shared by
prepare_fusion.py (training data) and schema_fusion_pt.py (--ask inference).

Why: greedy-decode eval showed the model nails schema/jargon retrieval (the
copy head) but loses ~24 points of exact-match transcribing literal values --
unseen ids (ABC123, k1) shred into rare BPE pieces, and dates like
"Apr 2025" -> '2025-04-01' AND '2025-04-30' require *computing* month
boundaries, which is a job for code, not gradient descent.

How: literals detected in the question are replaced with slot tokens
[val1]..[val10] (BPE special tokens). The same values are replaced in the
target SQL, so the model learns to copy single slot tokens. At inference the
generated slots are substituted back with the real (case-preserved, date-
normalized) values. Selection stays learned; emission stays deterministic --
same philosophy as the schema/jargon memory bank.

Detected literal classes (kept deliberately tight):
  1. ISO dates  2025-01-31          -> one slot, verbatim
  2. Month-year "Apr 2025"          -> TWO slots: month start + month end
  3. Alphanumeric ids  OBD123, Q8CD -> one slot, verbatim (case preserved)
  4. Standalone integers (not 0/1)  -> one slot ("top 5", "7 days"; 0/1
     excluded because they collide with ProdOrder = 0 / @expand... = 1)
Date phrases without a year ("month of march") are NOT slotted -- the year is
not recoverable deterministically, so those stay literal.
"""
import re
import calendar
import difflib

MAX_SLOTS = 10

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
MONYR_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})\b",
    re.IGNORECASE)
# Month with NO year ("month of march") -> resolved to DEFAULT_YEAR by
# convention, like a production bot resolving to the current year. "may" is
# deliberately excluded (collides with the modal verb); explicit full/abbrev
# forms only, so e.g. "marine" can't match "mar".
DEFAULT_YEAR = 2025
MON_NOYR_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
    r"(?!\s*\d{4})",
    re.IGNORECASE)
# Contains at least one digit AND one letter (so plain numbers/words skip it).
ID_RE = re.compile(r"\b(?=[A-Za-z_0-9]*\d)(?=[A-Za-z_0-9]*[A-Za-z])[A-Za-z_0-9]{2,}\b")
INT_RE = re.compile(r"\b\d+\b")


def extract_slots(text, max_slots=MAX_SLOTS):
    """Returns (delexed_text, slot_values). slot_values[k] is the SQL-side
    string for token [val{k+1}]. A month-year span consumes two slots."""
    spans = []   # (start, end, [sql_value, ...])
    taken = []

    def overlaps(s, e):
        return any(not (e <= s0 or s >= e0) for s0, e0 in taken)

    for m in ISO_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        spans.append((m.start(), m.end(), [f"{y:04d}-{mo:02d}-{d:02d}"]))
        taken.append(m.span())
    for m in MONYR_RE.finditer(text):
        if overlaps(*m.span()):
            continue
        mo = _MONTHS[m.group(1).lower()[:3]]
        yr = int(m.group(2))
        last = calendar.monthrange(yr, mo)[1]
        spans.append((m.start(), m.end(),
                      [f"{yr:04d}-{mo:02d}-01", f"{yr:04d}-{mo:02d}-{last:02d}"]))
        taken.append(m.span())
    for m in MON_NOYR_RE.finditer(text):
        if overlaps(*m.span()):
            continue
        mo = _MONTHS[m.group(1).lower()[:3]]
        last = calendar.monthrange(DEFAULT_YEAR, mo)[1]
        spans.append((m.start(), m.end(),
                      [f"{DEFAULT_YEAR:04d}-{mo:02d}-01",
                       f"{DEFAULT_YEAR:04d}-{mo:02d}-{last:02d}"]))
        taken.append(m.span())
    for m in ID_RE.finditer(text):
        if overlaps(*m.span()):
            continue
        spans.append((m.start(), m.end(), [m.group(0)]))
        taken.append(m.span())
    for m in INT_RE.finditer(text):
        if overlaps(*m.span()) or m.group(0) in ("0", "1"):
            continue
        spans.append((m.start(), m.end(), [m.group(0)]))
        taken.append(m.span())

    spans.sort(key=lambda s: s[0])
    out, last, slot_values = [], 0, []
    for s, e, vals in spans:
        if len(slot_values) + len(vals) > max_slots:
            break
        out.append(text[last:s])
        toks = []
        for v in vals:
            slot_values.append(v)
            toks.append(f"[val{len(slot_values)}]")
        out.append(" " + " ".join(toks) + " ")
        last = e
    out.append(text[last:])
    return "".join(out), slot_values


def delex_output(out_str, slot_values):
    """Replace every occurrence of each slot's value in the target SQL with
    its slot token. Longest values first so e.g. OBD123 can't eat OBD12."""
    order = sorted(range(len(slot_values)), key=lambda i: -len(slot_values[i]))
    for i in order:
        pat = re.compile(r"(?<!\w)" + re.escape(slot_values[i]) + r"(?!\w)",
                         re.IGNORECASE)
        out_str = pat.sub(f" [val{i + 1}] ", out_str)
    return out_str


_REPAIR_TOK = re.compile(
    r"\[val\d+\]"                      # existing slot tokens (skipped)
    r"|\d{4}-\d{1,2}-\d{1,2}"          # residual ISO dates
    r"|(?=[A-Za-z_0-9]*\d)(?=[A-Za-z_0-9]*[A-Za-z])[A-Za-z_0-9]{2,}")  # residual ids


def _is_date(v):
    return bool(re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", v))


def repair_output(delexed_out, slot_values):
    """Repair residual quoted literals in the target by mapping them onto
    UNUSED input slots, instead of dropping the example.

    Repair direction is gold <- question (copy semantics): with slots, the
    emitted value must flow from what the user typed.
      - Residual id + a fuzzy-matching unused slot (typo'd question id like
        OB1D23 vs gold 'OBD123') -> gold id becomes that slot token.
      - Residual ISO date + unused date slots (gold has random/wrong dates
        for the question's "Jan 2025") -> mapped onto the question's date
        slots in order of appearance.
    Anything that can't be mapped stays put; the caller's residual check
    then decides to drop. Returns (repaired_text, n_repaired)."""
    used = {i for i in range(1, len(slot_values) + 1)
            if f"[val{i}]" in delexed_out}
    unused = [i for i in range(1, len(slot_values) + 1) if i not in used]
    if not unused:
        return delexed_out, 0
    unused_dates = [i for i in unused if _is_date(slot_values[i - 1])]
    unused_other = [i for i in unused if not _is_date(slot_values[i - 1])]
    n_rep = 0

    def fix_quoted(qm):
        nonlocal n_rep
        def fix_tok(m):
            nonlocal n_rep
            tok = m.group(0)
            if tok.startswith("[val"):
                return tok
            if _is_date(tok):
                if unused_dates:
                    k = unused_dates.pop(0)
                    n_rep += 1
                    return f" [val{k}] "
                return tok
            # id: best fuzzy match among unused non-date slot values
            best_k, best_r = None, 0.0
            for k in unused_other:
                r = difflib.SequenceMatcher(
                    None, tok.lower(), slot_values[k - 1].lower()).ratio()
                if r > best_r:
                    best_k, best_r = k, r
            if best_k is not None and best_r >= 0.6:
                unused_other.remove(best_k)
                n_rep += 1
                return f" [val{best_k}] "
            return tok
        return _REPAIR_TOK.sub(fix_tok, qm.group(0))

    repaired = re.sub(r"'[^']*'", fix_quoted, delexed_out)
    return repaired, n_rep


def relex(sql, slot_values):
    """Substitute generated slot tokens back with real values. Descending
    index so [val10] is replaced before [val1]."""
    for i in range(len(slot_values), 0, -1):
        sql = sql.replace(f"[val{i}]", slot_values[i - 1])
    # Unfilled slots (model emitted a slot the question didn't provide).
    sql = re.sub(r"\[val\d+\]", "?", sql)
    return sql
