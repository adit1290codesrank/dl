import os
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

def extract_for_transformer():
    print("--- 🧠 DATA PIPELINE: SPATIOTEMPORAL TRANSFORMER ---")

    # Your exact SQL Query
    query = text("""
    WITH Time_Skeleton AS (
        SELECT generate_series(
            CURRENT_DATE - INTERVAL '180 days', 
            CURRENT_DATE, 
            INTERVAL '1 hour'
        ) AS slot
    ),
    Node_Configs AS (
        SELECT DISTINCT node_type, location_id FROM Lab_Nodes
    ),
    Capacity AS (
        SELECT node_type, location_id, COUNT(*) as total_nodes
        FROM Lab_Nodes GROUP BY node_type, location_id
    ),
    Hourly_Usage AS (
        SELECT 
            ts.slot,
            nc.node_type,
            nc.location_id,
            COUNT(r.reservation_id) as active_bookings
        FROM Time_Skeleton ts
        CROSS JOIN Node_Configs nc
        LEFT JOIN Reservations r ON r.node_id IN (
            SELECT node_id FROM Lab_Nodes 
            WHERE node_type = nc.node_type AND location_id = nc.location_id
        )
        AND ts.slot >= r.start_time AND ts.slot < r.end_time
        GROUP BY ts.slot, nc.node_type, nc.location_id
    )
    SELECT 
        EXTRACT(ISODOW FROM h.slot) as day_of_week, -- 1=Mon, 7=Sun
        EXTRACT(HOUR FROM h.slot) as hour_of_day,
        h.location_id,
        h.node_type,
        LEAST(CAST(h.active_bookings AS FLOAT) / NULLIF(c.total_nodes, 0), 1.0) as utilization_rate
    FROM Hourly_Usage h
    JOIN Capacity c ON h.node_type = c.node_type AND h.location_id = c.location_id;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("❌ No data found.")
        return

    # 1. Base Encoding (Force everything to start at 0)
    df['hour_of_day'] = df['hour_of_day'].astype(int) # 0 to 23
    df['day_of_week'] = df['day_of_week'].astype(int) - 1 # ISODOW is 1-7, shift to 0-6
    
    le_loc = LabelEncoder()
    df['loc_enc'] = le_loc.fit_transform(df['location_id']) # 0 to L
    
    le_node = LabelEncoder()
    df['node_enc'] = le_node.fit_transform(df['node_type']) # 0 to N

    # 2. Apply Strict Transformer Offsets!
    OFFSET_HOUR = 0
    OFFSET_DAY = 24
    OFFSET_LOC = OFFSET_DAY + 7
    OFFSET_NODE = OFFSET_LOC + len(le_loc.classes_)
    TOTAL_VOCAB = OFFSET_NODE + len(le_node.classes_)

    df['hour_idx'] = df['hour_of_day'] + OFFSET_HOUR
    df['day_idx'] = df['day_of_week'] + OFFSET_DAY
    df['loc_idx'] = df['loc_enc'] + OFFSET_LOC
    df['node_idx'] = df['node_enc'] + OFFSET_NODE

    # 3. Export clean sequence data for C++ Engine
    export_df = df[['hour_idx', 'day_idx', 'loc_idx', 'node_idx', 'utilization_rate']]
    export_df.to_csv("transformer_data.csv", index=False)

    # 4. Save the artifact (You need the Encoders + Offsets for inference later!)
    artifact = {
        "loc_encoder": le_loc, 
        "node_encoder": le_node,
        "offsets": {"hour": OFFSET_HOUR, "day": OFFSET_DAY, "loc": OFFSET_LOC, "node": OFFSET_NODE},
        "total_vocab": TOTAL_VOCAB
    }
    joblib.dump(artifact, "transformer_metadata.pkl")
    
    print(f"✅ Exported {len(df)} rows to CSV.")
    print(f"✅ TOTAL_VOCAB_SIZE required for C++: {TOTAL_VOCAB}")

if __name__ == "__main__":
    extract_for_transformer()