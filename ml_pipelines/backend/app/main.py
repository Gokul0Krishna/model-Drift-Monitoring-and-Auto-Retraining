from fastapi import FastAPI, BackgroundTasks, Depends
import mlflow.pyfunc
import pandas as pd
from app.tasks import trigger_auto_retraining
from ml.monitoring import check_data_drift
# Assuming SQLAlchemy infrastructure configured in database.py
from app.database import engine, PredictionRecord 
from sqlalchemy.orm import Session

app = FastAPI(title="Production MLOps Serving Engine")

# Load production model dynamically from MLflow registry on startup
# 'models:/production_model/latest' matches your registered name
try:
    model = mlflow.pyfunc.load_model("models:/production_model/latest")
except Exception:
    model = None # Handle fallback if no model is registered on first run

@app.post("/predict")
async def predict(features: dict, db: Session = Depends(get_db)):
    # 1. Convert feature dict to dataframe for inference
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    
    # 2. Log incoming feature arrays asynchronously to PostgreSQL for drift checking
    record = PredictionRecord(features=features, prediction=int(prediction))
    db.add(record)
    db.commit()
    
    return {"prediction": int(prediction)}

@app.post("/monitor/check-drift")
async def check_drift_endpoint(background_tasks: BackgroundTasks):
    # Fetch data sets from your DB layers
    reference_df = pd.read_csv("ml/reference.csv")
    current_df = get_all_production_data()
    
    drift_detected = check_data_drift(reference_df, current_df)
    
    if drift_detected:
        # Trigger training async without freezing the API response
        trigger_auto_retraining.delay()
        return {"status": "Drift Detected", "action": "Retraining triggered via Celery"}
        
    return {"status": "No Drift Detected", "action": "None"}