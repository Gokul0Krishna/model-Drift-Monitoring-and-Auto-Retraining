import pandas as pd
import requests
import time

BASE_URL = "http://localhost:8000"
CSV_PATH = "model_drift_shipping_pipeline/data/test_set.csv"
BATCH_SIZE = 1000

# Map CSV columns to the schema field names expected by the API
COLUMN_MAP = {
    "Warehouse_block": "warehouse_block",
    "Mode_of_Shipment": "mode_of_shipment",
    "Customer_care_calls": "customer_care_calls",
    "Customer_rating": "customer_rating",
    "Cost_of_the_Product": "cost_of_the_product",
    "Prior_purchases": "prior_purchases",
    "Product_importance": "product_importance",
    "Gender": "gender",
    "Discount_offered": "discount_offered",
    "Weight_in_gms": "weight_in_gms",
}

def main():
    df = pd.read_csv(CSV_PATH)
    # Keep only the columns the API expects (drop ID and target)
    df = df.rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())]

    total_rows = len(df)
    print(f"Total rows to ingest: {total_rows}")

    for start in range(0, total_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_rows)
        batch_df = df.iloc[start:end]
        records = batch_df.to_dict(orient="records")

        print(f"\n--- Sending batch {start // BATCH_SIZE + 1}  (rows {start} – {end - 1}) ---")

        # 1) Send to /ingest
        resp = requests.post(
            f"{BASE_URL}/ingest",
            json={"records": records},
            timeout=120,
        )

        if resp.status_code != 200:
            print(f"  [ERROR] /ingest returned {resp.status_code}: {resp.text}")
            continue

        data = resp.json()
        analysis_id = data.get("analysis_id")
        batch_range = data.get("batch_range", {})
        start_id = batch_range.get("start_id")
        end_id = batch_range.get("end_id")

        print(f"  Ingested {data.get('records_ingested')} records  |  "
              f"DB IDs {start_id}–{end_id}  |  analysis_id={analysis_id}")

        # 2) If analysis_id is truthy, trigger retraining
        if analysis_id:
            print(f"  Drift detected (analysis_id={analysis_id}). Triggering /retrain_model ...")
            retrain_resp = requests.post(
                f"{BASE_URL}/retrain_model",
                params={"start_id": start_id, "end_id": end_id},
                timeout=300,
            )
            if retrain_resp.status_code == 200:
                print(f"  [RETRAIN] {retrain_resp.json()}")
            else:
                print(f"  [RETRAIN ERROR] {retrain_resp.status_code}: {retrain_resp.text}")
        else:
            print("  No drift detected — skipping retrain.")

        # Small pause between batches to avoid overwhelming the server
        time.sleep(1)

    print("\n=== All batches processed ===")


if __name__ == "__main__":
    main()
