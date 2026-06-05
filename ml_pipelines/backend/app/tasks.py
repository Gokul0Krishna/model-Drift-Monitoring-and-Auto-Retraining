from celery import Celery
import os
import pandas as pd
from ml.pipeline import train_and_register_model
from database import get_all_production_data # Helper to fetch from Postgres

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task
def trigger_auto_retraining():
    print("🤖 Automated Retraining Worker Initiated...")
    
    # 1. Fetch baseline data & production data collected so far
    reference_df = pd.read_csv("ml/reference.csv")
    current_df = get_all_production_data() 
    
    # Ensure you have enough new data to justify training
    if len(current_df) < 500:
        return "Not enough new data samples to retrain."
        
    # 2. Re-combine and create updated train/val splits
    # For simplicity, combine reference and new production data
    combined_df = pd.concat([reference_df, current_df]).drop_duplicates()
    
    # Split
    train_df = combined_df.sample(frac=0.8, random_state=42)
    val_df = combined_df.drop(train_df.index)
    
    # 3. Train new version
    model_uri, new_auc = train_and_register_model(train_df, val_df)
    
    # 4. Optional: Compare against current active version before promoting
    # For now, MLflow bumps the version automatically in the Model Registry.
    return f"Success! New model registered at {model_uri} with AUC: {new_auc}"