"""
generate_paraphrase_dataset.py -- paraphrase-rich dataset generator.

Why: --ask probing showed the model is template-matching: questions phrased
exactly like training rows produce perfect SQL, any rephrasing collapses into
random schema picks. The old generator's "paraphrases" were five trivial
lambdas + typo noise, so the model never learned phrasing invariance -- and a
from-scratch model has no pretraining to fall back on. Language robustness
must come from the data.

What: for each seed query family (from all_samples.json), a library of
genuinely different human phrasings x varied values (warehouses, SMUs,
transporters, ids, months) substituted consistently into question and SQL.

Honest split: the LAST 2 question templates of every family are reserved for
validation only. Val therefore contains phrasings the model has never seen,
making eval measure paraphrase generalization instead of template recall.

Output: data/synthetic_dataset.json (train block first, then val block;
prepare_fusion.py's sequential 80/20 split lands exactly on the boundary).
The previous dataset is backed up once to data/synthetic_dataset_orig.json.
"""
import json
import os
import random
import calendar
import shutil

random.seed(7)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "data", "synthetic_dataset.json")

N_TRAIN = 10400
N_VAL = 2600
VAL_TEMPLATES_PER_FAMILY = 2  # last K question templates are val-only

# Carrier augmentation, TRAIN ONLY (val phrasings stay fixed so the benchmark
# is comparable across runs). Teaches invariance to politeness/filler tokens
# around the actual request -- the residual failure mode is novel carriers
# collapsing near-twin choices to the more frequent twin.
PREPENDS = ["", "", "", "", "please ", "hey ", "can you ", "quickly ",
            "i need ", "pls ", "could you "]
APPENDS = ["", "", "", "", " please", " thanks", " ?", " right away", " asap"]


def augment_carrier(q):
    return random.choice(PREPENDS) + q + random.choice(APPENDS)

# Schema (INCLUDE_IN_MODEL only -- the memory bank can only emit these) for
# the GENERAL T-SQL families: they teach compositional retrieval ("question
# names a column -> point at its memory row") over all ~400 columns, so the
# model can answer arbitrary queries instead of only the business templates.
with open(os.path.join(BASE, "all_tables.json"), encoding="utf-8") as f:
    _schema = json.load(f)
TABLE_COLS = {}
for _it in _schema:
    if _it.get("INCLUDE_IN_MODEL") and _it.get("COLUMN_NAME"):
        TABLE_COLS.setdefault(_it["TABLE_NAME"], []).append(_it["COLUMN_NAME"])
GEN_TABLES = [t for t, cs in TABLE_COLS.items() if len(cs) >= 3]

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------
WAREHOUSES = ["Bangalore", "Cochin", "Chennai", "Mumbai", "Delhi", "Hyderabad",
              "Kolkata", "Pune", "Indore", "Nagpur"]
PLANTS = ["Q8CD", "Q80D", "M20B", "Q81D", "P10X", "K77A", "R44T"]
SMUS = ["Marine", "Protective Coating", "Deco", "Coatings", "Powder"]
TRANSPORTERS = ["SAFE EXPRESS", "CUBE LOGISTICS", "SYNERGY BAXIS", "GATI",
                "DELHIVERY", "TCI EXPRESS", "VRL LOGISTICS", "BLUE DART"]
BIZ = ["Deco", "Coatings", "Marine", "Powder"]
MONTH_NAMES = ["January", "February", "March", "April", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"])}
FIELD_SETTERS = [
    ("Transporter", "set transporter as {tr}", "'{tr}'"),
    ("PickListEmailDate", "set OBD date as {d1}", "'{d1}'"),
    ("InvoiceDate", "set invoice date as {d1}", "'{d1}'"),
    ("ActDeliveryDate", "set actual delivery date as {d1}", "'{d1}'"),
    ("MtrlMvdFromFctryDate", "set dispatch date as {d1}", "'{d1}'"),
]


def rand_id():
    return (random.choice(["OBD", "PL", "SO", "AX", "KK", "ZN", ""])
            + "".join(random.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=random.randint(1, 3)))
            + "".join(random.choices("0123456789", k=random.randint(2, 4))))


def rand_iso(y0=2023, y1=2026):
    y = random.randint(y0, y1)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def rand_month():
    """Returns (text, start_iso, end_iso) e.g. ('April 2025', ...)."""
    name = random.choice(MONTH_NAMES)
    yr = random.randint(2023, 2026)
    mo = MONTH_NUM[name]
    last = calendar.monthrange(yr, mo)[1]
    text = random.choice([f"{name} {yr}", f"{name[:3]} {yr}",
                          f"the month of {name} {yr}"])
    return text, f"{yr:04d}-{mo:02d}-01", f"{yr:04d}-{mo:02d}-{last:02d}"


def fill(d):
    """Random value assignment shared by question and SQL templates."""
    mon_text, m1, m2 = rand_month()
    d1, d2 = sorted([rand_iso(), rand_iso()])
    ids = [rand_id() for _ in range(random.randint(2, 4))]
    # general-query fields: a random table + columns from THAT table
    t = random.choice(GEN_TABLES)
    cols = TABLE_COLS[t]
    date_cols = [c for c in cols if "date" in c.lower()] or cols
    ordw, ordsql = random.choice([("ascending", "ASC"), ("descending", "DESC")])
    v = random.choice(WAREHOUSES + TRANSPORTERS
                      + ["Delivered", "Cancelled", "Pending Picking",
                         "Pending Dispatch", "Sales", "Stock Transfer"]
                      + [rand_id()])
    return {
        "smu": random.choice(SMUS), "wh": random.choice(WAREHOUSES),
        "plant": random.choice(PLANTS), "tr": random.choice(TRANSPORTERS),
        "biz": random.choice(BIZ), "id": rand_id(),
        "ids_lines": "\n".join(ids), "ids_csv": ",".join(ids),
        "mon": mon_text, "m1": m1, "m2": m2,
        "d1": d1, "d2": d2,
        "n": random.choice([3, 5, 10, 15, 20, 25, 50]),
        "days": random.choice([3, 5, 7, 10, 14, 30]),
        "t": t, "c": random.choice(cols), "c_sel": random.choice(cols),
        "c2": random.choice(cols), "c_date": random.choice(date_cols),
        "cj1": random.choice(TABLE_COLS["AN_CUSTOMER_VS_ASM_RSM"]),
        "cj2": random.choice(TABLE_COLS["AN_LOGISTICS_TRACKER"]),
        "agg": random.choice(["COUNT", "SUM", "AVG", "MIN", "MAX"]),
        "ordw": ordw, "ord": ordsql,
        "num": random.choice([5, 10, 50, 100, 500, 1000]),
        "v": v,
        **d,
    }


# ---------------------------------------------------------------------------
# Families: (weight, [question templates...], sql template)
# The last VAL_TEMPLATES_PER_FAMILY question templates are validation-only.
# ---------------------------------------------------------------------------
FAMILIES = [
    # --- MPY sales count -------------------------------------------------
    (4, [
        "What is the number of sales for MPY",
        "How many sales were made for MPY",
        "Count the sales for MPY",
        "MPY sales count please",
        "give me the total sales for MPY",
        "how many MPY sales do we have",
        "How many sales were made for MPY in total?",
        "what's the MPY sales number",
    ], "SELECT COUNT(*)\nFROM AN_LOGISTICS_TRACKER\nWHERE SMU IN( 'Protective Coating', 'Marine')\nAND NatureOfTransaction = 'Sales'"),

    # --- count OBDs per SMU ----------------------------------------------
    (5, [
        "Count the number of OBDs for {smu}",
        "How many OBDs are there for {smu}",
        "number of OBDs for the {smu} SMU",
        "Count OBDs where the business unit is {smu}",
        "how many OBDs does {smu} have",
        "Count the OBDs for the {smu} business unit",
        "total OBD count for {smu} please",
    ], "SELECT COUNT(*)\nFROM AN_LOGISTICS_TRACKER\nWHERE SMU = '{smu}'"),

    # --- MPY sales count within a month (composite: jargon + date range) ---
    (3, [
        "How many sales were made for MPY in {mon}",
        "What is the number of sales for MPY in {mon}",
        "MPY sales count for {mon}",
        "count the MPY sales during {mon}",
        "total MPY sales in {mon}",
        "how many MPY sales happened in {mon}",
        "number of sales for MPY during {mon}",
    ], "SELECT COUNT(*)\nFROM AN_LOGISTICS_TRACKER\nWHERE SMU IN( 'Protective Coating', 'Marine')\nAND NatureOfTransaction = 'Sales'\nAND InvoiceDate BETWEEN '{m1}' AND '{m2}'"),

    # --- single-date threshold queries (the "after DATE" gap) ---------------
    (3, [
        "List all OBDs dispatched after {d1}",
        "OBDs dispatched after {d1}",
        "which OBDs were dispatched after {d1}",
        "show OBDs with dispatch date after {d1}",
        "give me every OBD dispatched since {d1}",
        "OBDs that got dispatched after {d1}",
        "list the OBDs dispatched later than {d1}",
    ], "SELECT PickListId\nFROM AN_LOGISTICS_TRACKER\nWHERE MtrlMvdFromFctryDate >= '{d1}'"),
    (3, [
        "List all OBDs invoiced after {d1}",
        "OBDs invoiced after {d1}",
        "which OBDs were invoiced after {d1}",
        "show OBDs with invoice date after {d1}",
        "OBDs billed after {d1}",
        "give me every OBD invoiced since {d1}",
        "list the OBDs invoiced later than {d1}",
    ], "SELECT PickListId\nFROM AN_LOGISTICS_TRACKER\nWHERE InvoiceDate >= '{d1}'"),

    # --- distinct SSM per SMU (SSM had no business family of its own) -------
    (3, [
        "Who are the SSM for {smu}",
        "List the SSMs for the {smu} business unit",
        "which SSMs handle {smu}",
        "SSM list for {smu}",
        "show me all SSM working on {smu}",
        "who are the senior sales managers of {smu}",
    ], "SELECT DISTINCT SSM\nFROM AN_CUSTOMER_VS_ASM_RSM\nWHERE SMU = '{smu}'"),

    # --- distinct RSM per SMU ---------------------------------------------
    (5, [
        "Who are the RSM for {smu}",
        "List the RSMs for the {smu} business unit",
        "Which RSMs handle {smu}",
        "regional sales managers for {smu}",
        "show me all RSM working on {smu}",
        "who are the regional sales managers of {smu}",
        "RSM list for {smu} please",
    ], "SELECT DISTINCT RSM\nFROM AN_CUSTOMER_VS_ASM_RSM\nWHERE SMU = '{smu}'"),

    # --- ASM for customer code ---------------------------------------------
    (5, [
        "Who is the ASM for customer code {id}",
        "Who is the Area Sales Manager for customer code {id}",
        "ASM of customer {id}",
        "tell me the ASM for customer {id}",
        "which ASM handles customer code {id}",
        "who handles customer {id} as ASM",
        "name the ASM assigned to customer code {id}",
        "what salesperson is the ASM of customer {id}",
        "Which salesperson is the ASM for customer code {id}?",
        "find the area sales manager for customer {id}",
    ], "SELECT SalesPersonName\nFROM AN_CUSTOMER_VS_ASM_RSM\nWHERE CustomerCode = '{id}'"),

    # --- SONum for OBD ------------------------------------------------------
    (5, [
        "What is the Sales Order Number for OBD {id}",
        "Sales order number of OBD {id}",
        "get the SO number for OBD {id}",
        "which sales order does OBD {id} belong to",
        "SONum for OBD {id} please",
        "what is the SO for OBD {id}",
        "sales order for OBD number {id}",
        "give me the sales order number against OBD {id}",
        "SO number of the OBD {id}",
        "fetch the sales order number for OBD {id}",
        "tell me the SO number linked with OBD {id}",
        "i want the sales order number of OBD {id}",
        "OBD {id} belongs to which sales order",
        "Find the sales order number for OBD number {id}",
        "what SO is linked to OBD {id}",
    ], "SELECT SONum\nFROM AN_LOGISTICS_TRACKER\nWHERE PickListID = '{id}'"),

    # --- OBDs invoiced in month ---------------------------------------------
    (5, [
        "Give all the OBDs invoiced in {mon}",
        "List the OBDs invoiced during {mon}",
        "which OBDs were invoiced in {mon}",
        "show OBDs billed in {mon}",
        "OBDs with invoice date in {mon}",
        "List all OBDs that got invoiced in {mon}",
        "fetch every OBD invoiced in {mon}",
    ], "SELECT PickListId\nFROM AN_LOGISTICS_TRACKER\nWHERE InvoiceDate BETWEEN '{m1}' AND '{m2}'"),

    # --- distinct SONum count in month ---------------------------------------
    (3, [
        "How many different SONum were Invoiced during {mon}?",
        "Count of distinct sales orders invoiced in {mon}",
        "how many unique SO numbers were invoiced in {mon}",
        "distinct SONum count for {mon}",
        "How many different sales orders got invoiced in {mon}",
        "unique sales order count invoiced during {mon}",
    ], "SELECT COUNT(DISTINCT SONum)\nFROM AN_LOGISTICS_TRACKER\nWHERE InvoiceDate BETWEEN '{m1}' AND '{m2}'"),

    # --- top customers by volume ---------------------------------------------
    (4, [
        "Top {n} Customer based on volume invoiced in {mon} excluding Stock Transfers?",
        "top {n} customers by invoiced volume in {mon} without stock transfers",
        "Who are the top {n} customers on volume invoiced in {mon}, excluding stock transfer",
        "best {n} customers by volume invoiced for {mon} excluding stock transfers",
        "who are our top {n} customers by invoiced volume in {mon} excluding stock transfers",
        "rank the top {n} customers on volume invoiced during {mon}, ignore stock transfers",
        "top {n} customers for {mon} by volume, no stock transfer",
        "List the top {n} customers based on volume invoiced in {mon} excluding Stock Transfers",
        "give top {n} customers by VolumeInvoiced during {mon}, no stock transfers",
    ], "SELECT TOP {n} ShipCustomerName\nFROM AN_LOGISTICS_TRACKER\nWHERE InvoiceDate BETWEEN '{m1}' AND '{m2}'\nAND NatureOfTransaction != 'Stock Transfer'\nGROUP BY ShipCustomerName\nORDER BY SUM(VolumeInvoiced) DESC"),

    # --- business units from warehouse -----------------------------------------
    (4, [
        "What are the different business units from {wh} Warehouse?",
        "List the SMUs operating out of {wh} warehouse",
        "which business units ship from {wh}",
        "distinct business units in the {wh} warehouse",
        "which SMUs operate from the {wh} warehouse",
        "business units present at {wh}",
        "show the SMU list for {wh} warehouse",
        "What business units do we have at {wh}?",
    ], "SELECT DISTINCT SMU\nFROM AN_LOGISTICS_TRACKER\nWHERE SiteId = '{wh}'"),

    # --- RSM for OBD (join) -------------------------------------------------
    (4, [
        "Who is the RSM for OBD Number {id}",
        "RSM for OBD {id}",
        "which RSM owns OBD number {id}",
        "find the RSM responsible for OBD {id}",
        "Who is the regional sales manager for OBD {id}",
        "tell me the RSM behind OBD number {id}",
    ], "SELECT A.RSM,B.PickListId\nFROM AN_CUSTOMER_VS_ASM_RSM A JOIN AN_LOGISTICS_TRACKER B\nON A.CustomerCode = B.SoldToCustomerId\nWHERE B.PickListId = '{id}'"),

    # --- track vehicle --------------------------------------------------------
    (4, [
        "Track the vehicle with OBD Number {id}",
        "track vehicle for OBD {id}",
        "where is the vehicle for OBD number {id}",
        "vehicle tracking for OBD {id}",
        "Track the truck carrying OBD {id}",
        "locate the vehicle for OBD number {id}",
    ], "EXEC VehicleTracker_GetInfo @pickListId = '{id}'"),

    # --- vehicle utilization (period) ------------------------------------------
    (4, [
        "Vehicle Utilization for the period between {d1} to {d2}",
        "What is the overall vehicle utilization between {d1} and {d2}?",
        "vehicle utilization from {d1} to {d2}",
        "show vehicle utilization for {d1} - {d2}",
        "How was the vehicle utilization between {d1} and {d2}",
        "overall vehicle utilisation for the window {d1} to {d2}",
    ], "EXEC ANDashBoardVehicleUtilization @startDate = '{d1}', @endDate = '{d2}'"),

    # --- vehicle utilization (month, by type/transporter) ------------------------
    (3, [
        "What is the vehicle utilization by vehicle type in {mon}?",
        "vehicle utilization by type for {mon}",
        "Show me the vehicle utilization by vehicle type for {mon}",
        "break vehicle utilization down by vehicle type for {mon}",
        "vehicle utilisation per vehicle type in {mon}",
    ], "EXEC ANDashBoardVehicleUtilization @startDate = '{m1}', @endDate = '{m2}', @expandVehicleType = 1"),
    (3, [
        "What is the vehicle utilization by transporter in {mon}?",
        "vehicle utilization by transporter for {mon}",
        "Show me the vehicle utilization for {mon}",
        "list the transporter wise vehicle utilization for {mon}",
        "vehicle utilisation per transporter in {mon}",
    ], "EXEC ANDashBoardVehicleUtilization @startDate = '{m1}', @endDate = '{m2}', @expandTransporter = 1"),
    (3, [
        "List vehicle utilization for LR no {id}",
        "vehicle utilization for LR number {id}",
        "vehicle utilization of the LR {id}",
        "LR no {id} vehicle utilization",
        "what is the vehicle utilization for LR number {id}",
        "check vehicle utilization for the LR no {id}",
        "utilization for LR {id}",
        "show vehicle utilisation of LR no {id}",
    ], "EXEC ANDashBoardVehicleUtilization @transporterlrno='{id}'"),

    # --- transporter dashboard ----------------------------------------------
    (3, [
        "Vendor or Transporter Dashboard for {tr} for {mon}",
        "Transporter dashboard for {tr} in {mon}",
        "show the vendor dashboard of {tr} for {mon}",
        "open the transporter dashboard for {tr} covering {mon}",
        "{tr} transporter dashboard for {mon}",
    ], "EXEC Report_SingleTransporterDashboard '_CHATBOT_', 'deb', '{tr}', '{m1}', '{m2}';\nSELECT *\nFROM REPORT_SINGLE_TRANSPORTER_DASHBOARD"),
    (3, [
        "What is the DOT (Delivery On Time) for {tr} for {mon}",
        "DOT percentage of {tr} in {mon}",
        "how was the delivery on time for {tr} during {mon}",
        "delivery on time percentage for {tr} in {mon}",
        "show me the DOT of {tr} for {mon}",
        "Delivery On Time for {tr}, {mon}",
        "what's the DOT for {tr} in {mon}",
    ], "EXEC Report_SingleTransporterDashboard '_CHATBOT_', 'deb', '{tr}', '{m1}', '{m2}'\nSELECT NumberOfConsignments, DOTPercentage\nFROM REPORT_SINGLE_TRANSPORTER_DASHBOARD"),
    (3, [
        "What is the average time taken to upload PODs for {tr} for {mon}",
        "average POD upload days for {tr} in {mon}",
        "how long does {tr} take to upload PODs, for {mon}",
        "POD upload time of {tr} during {mon}",
        "avg days to upload PODs for {tr} in {mon}",
    ], "EXEC Report_SingleTransporterDashboard '_CHATBOT_', 'deb', '{tr}', '{m1}', '{m2}'\nSELECT PODUploadCount, AvgUploadDays\nFROM REPORT_SINGLE_TRANSPORTER_DASHBOARD"),

    # --- OBD availability ------------------------------------------------------
    (3, [
        "Why following OBDs are not available?\n{ids_lines}",
        "Check the availability of the following OBDs\n{ids_lines}",
        "are these OBDs available\n{ids_lines}",
        "availability check for these OBDs\n{ids_lines}",
        "why can't I see these OBDs\n{ids_lines}",
    ], "EXEC OBDAvailability '_VIEWING_', '', '{ids_csv}'"),

    # --- mass update: field setters ----------------------------------------------
    (6, [
        "{setter} for the following OBDs\n{ids_lines}",
        "{setter} for these OBDs\n{ids_lines}",
        "for the OBDs below, {setter}\n{ids_lines}",
        "{setter} on all of these OBDs\n{ids_lines}",
        "kindly {setter} for the OBD list\n{ids_lines}",
        "{setter} for all these OBDs\n{ids_lines}",
        "go ahead and {setter} for the OBDs\n{ids_lines}",
        "need to {setter} for the following\n{ids_lines}",
        "{setter}, OBDs listed below\n{ids_lines}",
        "please {setter} on the following OBDs\n{ids_lines}",
        "{setter} for the listed OBDs\n{ids_lines}",
    ], "EXEC CB_OBDMassUpdate '_CB_USERNAME_', '{ids_csv}', '{field}', {value}"),
    (2, [
        "Reset transporters for the following OBDs\n{ids_lines}",
        "reset the transporter on these OBDs\n{ids_lines}",
        "unset the transporter for these OBDs\n{ids_lines}",
        "clear transporter for the OBDs below\n{ids_lines}",
        "remove the transporter assignment for these OBDs\n{ids_lines}",
    ], "EXEC CB_OBDMassUpdate '_CB_USERNAME_', '{ids_csv}', 'Transporter', NULL"),
    (2, [
        "Cancel following OBDs\n{ids_lines}",
        "cancel these OBDs\n{ids_lines}",
        "void the following OBDs\n{ids_lines}",
        "please cancel the OBDs below\n{ids_lines}",
        "mark the following OBDs as cancelled\n{ids_lines}",
    ], "EXEC CB_OBDMassUpdate '_CB_USERNAME_', '{ids_csv}', 'Cancelled', 1"),
    (2, [
        "Clear Proof Of Delivery for the following LRs\n{ids_lines}",
        "Remove PODs of the listed OBDs\n{ids_lines}",
        "remove the proof of delivery for these OBDs\n{ids_lines}",
        "wipe the POD records of the following OBDs\n{ids_lines}",
        "clear POD documents for the OBDs below\n{ids_lines}",
        "delete the POD for these OBDs\n{ids_lines}",
        "clear the PODs on the following OBDs\n{ids_lines}",
    ], "EXEC CB_OBDMassUpdate '_CB_USERNAME_', '{ids_csv}', 'PODDocId', ''"),
    (2, [
        "Mark these as {biz} OBDs\n{ids_lines}",
        "set business as {biz} for the following OBDs\n{ids_lines}",
        "flag the following OBDs under {biz}\n{ids_lines}",
        "tag the OBDs below as {biz}\n{ids_lines}",
        "these OBDs belong to {biz}, mark them\n{ids_lines}",
    ], "EXEC CB_OBDMassUpdate '_CB_USERNAME_', '{ids_csv}', 'Business', '{biz}'"),

    # --- pending dispatch / statuses ------------------------------------------
    (3, [
        "List top Warehouses where OBDs are Pending Dispatch for more than {days} days",
        "warehouses with OBDs pending dispatch over {days} days",
        "which warehouses have OBDs stuck in pending dispatch for {days}+ days",
        "list warehouses where dispatch is pending more than {days} days",
        "warehouses having OBDs in pending dispatch older than {days} days",
        "top sites where OBDs are pending dispatch beyond {days} days",
        "show warehouses with dispatch pending for more than {days} days",
    ], "SELECT SiteId, COUNT(PickListId)\nFROM AN_LOGISTICS_TRACKER\nWHERE InvoiceDate < DATEADD(DAY, -{days}, GETDATE())\nAND PendingStatus = 'Pending Dispatch'\nAND ProdOrder = 0\nGROUP BY SiteId\nORDER BY COUNT(*) DESC"),
    (3, [
        "List different pending statuses in {wh} warehouse for OBDs loaded after {mon}",
        "pending status breakdown for {wh} for OBDs after {mon}",
        "what pending statuses exist in {wh} for OBDs received after {mon}",
        "different pending statuses for {wh} OBDs loaded after {mon}",
        "list the pending status distribution in {wh} for OBDs after {mon}",
        "what are the pending statuses in the {wh} warehouse since {mon}",
        "show the pending statuses at {wh} warehouse since {mon}",
        "pending status counts in {wh} for OBDs loaded after {mon}",
    ], "SELECT PendingStatus, COUNT(PickListId)\nFROM AN_LOGISTICS_TRACKER\nWHERE PendingStatus NOT IN ('Cancelled', 'Delivered')\nAND ProdOrder = 0\nAND SiteId = '{wh}'\nAND PickListEmailDate >= '{m1}'\nGROUP BY PendingStatus\nORDER BY COUNT(*) DESC"),

    # --- time analysis ----------------------------------------------------------
    (3, [
        "List top vehicle types with highest time taken for OBD Create To Picking for {wh} for OBD received after {mon}",
        "vehicle types slowest at OBD create to picking in {wh} since {mon}",
        "which vehicle types take longest for OBD Create To Picking at {wh} after {mon}",
        "top vehicle types by OBD create-to-picking time for {wh}, OBDs after {mon}",
        "slowest vehicle types for create to picking in {wh} since {mon}",
    ], "SELECT VehicleType, AVG(dbo.HoursDifferenceF(PckLstFwdToWHDate,PckLstFwdToWHTime,PickListRetForInvDate,PickListRetForInvTime)) OBDCreateToPicking\nFROM AN_LOGISTICS_TRACKER\nWHERE PendingStatus NOT IN ('Cancelled', 'Pending Picking')\nAND ProdOrder = 0\nAND SiteId = '{wh}'\nAND PickListEmailDate >= '{m1}'\nGROUP BY VehicleType\nORDER BY OBDCreateToPicking DESC"),
    (3, [
        "List top destinations with highest time taken for Picking To Invoice for plant {plant} for OBD received after {mon}",
        "destinations slowest from picking to invoice for plant {plant} since {mon}",
        "which destinations take longest for Picking To Invoice at plant {plant} after {mon}",
        "top destinations by picking-to-invoice time for {plant}, OBDs after {mon}",
        "slowest destinations for picking to invoice at plant {plant} since {mon}",
    ], "SELECT ISNULL(ShipToDestinationOverride, ShipToDestination) Destination, \nAVG(dbo.HoursDifferenceF(PckLstFwdToWHDate,PckLstFwdToWHTime,PickListRetForInvDate,PickListRetForInvTime)) PickingToInvoice\nFROM AN_LOGISTICS_TRACKER\nWHERE PendingStatus NOT IN ('Cancelled', 'Pending Picking')\nAND ProdOrder = 0\nAND FromPlant = '{plant}'\nAND PickListEmailDate >= '{m1}'\nGROUP BY ISNULL(ShipToDestinationOverride, ShipToDestination)\nORDER BY PickingToInvoice DESC"),
    (5, [
        "List top {n} OBDs with most time taken for Invoicing to Dispatch for plant {plant} for OBD received after {mon}",
        "top {n} OBDs slowest from invoicing to dispatch at plant {plant} since {mon}",
        "which {n} OBDs took longest for Invoice to Dispatch for plant {plant} after {mon}",
        "top {n} OBDs by invoicing to dispatch delay for plant {plant} after {mon}",
        "invoicing to dispatch report, top {n} OBDs, plant {plant}, after {mon}",
        "invoice to dispatch delays, top {n} OBDs at plant {plant} since {mon}",
        "show the invoicing to dispatch report for plant {plant}, top {n}, after {mon}",
        "{n} worst OBDs on invoicing-to-dispatch time for {plant}, received after {mon}",
        "slowest {n} OBDs invoice to dispatch, plant {plant}, after {mon}",
    ], "SELECT TOP {n} PickListId OBDNo, ShipCustomerName, SiteId Warehouse, \nISNULL(ShipToDestinationOverride, ShipToDestination) Destination, \nPickListEmailDate, InvoiceDate, InvoiceTime,\nISNULL(MtrlMvdFromFctryDateOverride,MtrlMvdFromFctryDate) DispatchDate, \nISNULL(MtrlMvdFromFctryTimeOverride,MtrlMvdFromFctrytime) DispatchTime,\ndbo.HoursDifferenceF(InvoiceDate,InvoiceTime,ISNULL(MtrlMvdFromFctryDateOverride,MtrlMvdFromFctryDate),\nISNULL(MtrlMvdFromFctryTimeOverride,MtrlMvdFromFctrytime)) InvoiceToDispatchInMinutes\nFROM AN_LOGISTICS_TRACKER\nWHERE PendingStatus NOT IN ('Cancelled', 'Pending Picking')\nAND ProdOrder = 0\nAND FromPlant = '{plant}'\nAND PickListEmailDate >= '{m1}'\nORDER BY InvoiceToDispatchInMinutes DESC"),
    (5, [
        "List top {n} OBDs with most time taken for Picking to Dispatch for plant {plant} for OBD received after {mon}",
        "top {n} OBDs slowest from picking to dispatch at plant {plant} since {mon}",
        "which {n} OBDs took longest for Picking to Dispatch for plant {plant} after {mon}",
        "top {n} OBDs by picking to dispatch delay for plant {plant} after {mon}",
        "picking to dispatch report, top {n} OBDs, plant {plant}, after {mon}",
        "picking to dispatch delays, top {n} OBDs at plant {plant} since {mon}",
        "show the picking to dispatch report for plant {plant}, top {n}, after {mon}",
        "report the {n} slowest picking to dispatch OBDs for plant {plant} since {mon}",
        "{n} worst OBDs on picking-to-dispatch time for {plant}, received after {mon}",
        "slowest {n} OBDs picking to dispatch, plant {plant}, after {mon}",
    ], "SELECT TOP {n} PickListId OBDNo, ShipCustomerName, SiteId Warehouse, \nISNULL(ShipToDestinationOverride, ShipToDestination) Destination, \nPickListEmailDate OBDEmailDate, PickListRetForInvDate OBDRetForInvDate, PickListRetForInvTime OBDRetForInvTime,\nISNULL(MtrlMvdFromFctryDateOverride,MtrlMvdFromFctryDate) DispatchDate, \nISNULL(MtrlMvdFromFctryTimeOverride,MtrlMvdFromFctrytime) DispatchTime, \ndbo.HoursDifferenceF(PickListRetForInvDate,PickListRetForInvTime,ISNULL(MtrlMvdFromFctryDateOverride,MtrlMvdFromFctryDate),ISNULL(MtrlMvdFromFctryTimeOverride,MtrlMvdFromFctrytime)) PickingToDispatchInMinutes\nFROM AN_LOGISTICS_TRACKER\nWHERE PendingStatus NOT IN ('Cancelled', 'Pending Picking')\nAND ProdOrder = 0\nAND FromPlant = '{plant}'\nAND PickListEmailDate >= '{m1}'\nORDER BY PickingToDispatchInMinutes DESC"),

    # =======================================================================
    # GENERAL T-SQL families (over ALL schema tables/columns). These teach
    # compositional retrieval -- the question NAMES a column/table and the
    # model must point at the right memory row -- so arbitrary queries work,
    # not just the business templates above.
    # =======================================================================
    (10, [
        "Show me {c_sel} from {t} where {c} is {v}",
        "get {c_sel} from {t} where {c} = {v}",
        "list {c_sel} in {t} with {c} {v}",
        "{c_sel} from {t} for {c} {v}",
        "fetch {c_sel} of {t} where {c} equals {v}",
        "what is the {c_sel} in {t} when {c} is {v}",
        "select {c_sel} from {t} where {c} is {v}",
        "value of {c_sel} in {t} where {c} is {v}",
        "pull {c_sel} from {t} where the {c} equals {v}",
        "in table {t}, {c_sel} where {c} is {v}",
        "from {t} give me {c_sel} where {c} is {v}",
        "find {c_sel} for rows in {t} whose {c} is {v}",
    ], "SELECT {c_sel}\nFROM {t}\nWHERE {c} = '{v}'"),
    (4, [
        "Show all records from {t} where {c} is {v}",
        "everything in {t} with {c} {v}",
        "all rows of {t} where {c} = {v}",
        "dump {t} where {c} is {v}",
        "get all data from {t} for {c} {v}",
        "show me the full rows of {t} where {c} equals {v}",
    ], "SELECT *\nFROM {t}\nWHERE {c} = '{v}'"),
    (8, [
        "show top {n} {c_sel} from {t} ordered by {c} {ordw}",
        "top {n} {c_sel} in {t} by {c} {ordw}",
        "list the top {n} {c_sel} from {t} sorted by {c} {ordw}",
        "give me {n} {c_sel} from {t} ranked by {c} {ordw}",
        "first {n} {c_sel} of {t} ordered on {c} {ordw}",
        "{n} {c_sel} from {t}, order by {c} {ordw}",
        "top {n} {c_sel} sorted on {c} {ordw} from {t}",
        "from {t} fetch the top {n} {c_sel} by {c} {ordw}",
        "Show top {n} {c_sel} ordered by {c} {ordw} in {t}",
        "what are the top {n} {c_sel} in {t} by {c} {ordw}",
    ], "SELECT TOP {n} {c_sel}\nFROM {t}\nORDER BY {c} {ord}"),
    (8, [
        "{agg} of {c2} grouped by {c} in {t}",
        "show {agg} of {c2} per {c} from {t}",
        "what is the {agg} of {c2} for each {c} in {t}",
        "group {t} by {c} and give {agg} of {c2}",
        "compute {agg} on {c2} by {c} for {t}",
        "{agg} {c2} broken down by {c} in {t}",
        "per {c}, what is the {agg} of {c2} in {t}",
        "aggregate {c2} with {agg} grouped on {c} from {t}",
    ], "SELECT {c}, {agg}({c2})\nFROM {t}\nGROUP BY {c}"),
    (4, [
        "count of records per {c} in {t}",
        "how many rows per {c} in {t}",
        "record count grouped by {c} for {t}",
        "count rows in {t} for each {c}",
        "how many entries does {t} have per {c}",
        "number of records by {c} in {t}",
    ], "SELECT {c}, COUNT(*)\nFROM {t}\nGROUP BY {c}"),
    (6, [
        "{c_sel} from {t} where {c_date} between {d1} and {d2}",
        "show {c_sel} of {t} with {c_date} from {d1} to {d2}",
        "get {c_sel} in {t} where {c_date} is between {d1} and {d2}",
        "list {c_sel} from {t} for {c_date} in the range {d1} to {d2}",
        "{c_sel} of {t} where {c_date} falls between {d1} and {d2}",
        "fetch {c_sel} from {t}, {c_date} between {d1} and {d2}",
    ], "SELECT {c_sel}\nFROM {t}\nWHERE {c_date} BETWEEN '{d1}' AND '{d2}'"),
    (6, [
        "show {c_sel} from {t} where {c} is more than {num}",
        "{c_sel} in {t} with {c} greater than {num}",
        "get {c_sel} from {t} where {c} above {num}",
        "list {c_sel} of {t} where {c} exceeds {num}",
        "{c_sel} from {t} where {c} over {num}",
        "find {c_sel} in {t} having {c} larger than {num}",
    ], "SELECT {c_sel}\nFROM {t}\nWHERE {c} > {num}"),
    (6, [
        "{cj1} and {cj2} for customers joined with logistics where {cj2} is {v}",
        "join customer and logistics, show {cj1} and {cj2} where {cj2} = {v}",
        "show {cj1} with {cj2} from the customer-logistics join where {cj2} is {v}",
        "get {cj1} and {cj2} across AN_CUSTOMER_VS_ASM_RSM and AN_LOGISTICS_TRACKER where {cj2} is {v}",
        "customer {cj1} along with {cj2} where {cj2} equals {v}",
        "{cj1} and {cj2} joined on customer code where {cj2} is {v}",
    ], "SELECT A.{cj1}, B.{cj2}\nFROM AN_CUSTOMER_VS_ASM_RSM A JOIN AN_LOGISTICS_TRACKER B\nON A.CustomerCode = B.SoldToCustomerId\nWHERE B.{cj2} = '{v}'"),
    (3, [
        "{c} from {t}, if null use {v}",
        "show {c} of {t} defaulting nulls to {v}",
        "get {c} from {t} replacing null with {v}",
        "{c} in {t} with nulls shown as {v}",
        "select {c} from {t} but use {v} when null",
    ], "SELECT ISNULL({c}, '{v}')\nFROM {t}"),
    (3, [
        "difference in days between {c} and {c2} in {t}",
        "days between {c} and {c2} for {t}",
        "how many days between {c} and {c2} in {t}",
        "day gap from {c} to {c2} in {t}",
        "compute the day difference of {c} and {c2} for {t}",
    ], "SELECT DATEDIFF(day, {c}, {c2})\nFROM {t}"),
]


def render(family_idx, template, sql_template, aug=False):
    vals = fill({})
    if "{setter}" in template:
        field, setter_text, value_tpl = random.choice(FIELD_SETTERS)
        vals["field"] = field
        vals["setter"] = setter_text.format(**vals)
        vals["value"] = value_tpl.format(**vals)
    q = template.format(**vals)
    if aug:
        q = augment_carrier(q)
    sql = sql_template.format(**vals)
    return {"input": q, "output": sql}


def main():
    # one-time backup of the previous dataset
    orig = os.path.join(BASE, "data", "synthetic_dataset_orig.json")
    if os.path.exists(OUT) and not os.path.exists(orig):
        shutil.copyfile(OUT, orig)
        print(f"Backed up previous dataset -> {orig}")

    weights = [w for w, _, _ in FAMILIES]
    total_w = sum(weights)

    train, val = [], []
    for fi, (w, templates, sql) in enumerate(FAMILIES):
        n_tr = round(N_TRAIN * w / total_w)
        n_va = round(N_VAL * w / total_w)
        tr_templates = templates[:-VAL_TEMPLATES_PER_FAMILY]
        va_templates = templates[-VAL_TEMPLATES_PER_FAMILY:]
        for _ in range(n_tr):
            train.append(render(fi, random.choice(tr_templates), sql, aug=True))
        for _ in range(n_va):
            val.append(render(fi, random.choice(va_templates), sql))

    random.shuffle(train)
    random.shuffle(val)
    # train block first, then val: prepare_fusion's sequential int(n*0.8)
    # split must land EXACTLY on the boundary, or a seen-phrasing example
    # leaks into the held-out val block. Trim until they align.
    while int((len(train) + len(val)) * 0.8) != len(train):
        if int((len(train) + len(val)) * 0.8) > len(train):
            val.pop()
        else:
            train.pop()
    n_train = len(train)
    data = train + val

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)

    print(f"Wrote {len(data)} samples -> {OUT}")
    print(f"  train block: {n_train} (phrasings seen in training)")
    print(f"  val block:   {len(val)} (HELD-OUT phrasings -- last "
          f"{VAL_TEMPLATES_PER_FAMILY} templates of each family)")
    print(f"  families: {len(FAMILIES)}, total templates: "
          f"{sum(len(t) for _, t, _ in FAMILIES)}")
    print("NOTE: prepare_fusion.py must use the default sequential split "
          "(no --dedup reordering) so val stays template-held-out.")


if __name__ == "__main__":
    main()
