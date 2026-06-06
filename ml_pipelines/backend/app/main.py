from fastapi import FastAPI, BackgroundTasks, Depends
import mlflow.pyfunc
import pandas as pd
from app.tasks import trigger_auto_retraining
from ml.monitoring import check_data_drift
from app.database import get_db, get_all_production_data, engine
from sqlalchemy.orm import Session
from app.database import Base 
from app.model import PredictionLog, PredictionRequest
import logging
import time
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LIFESPAN STARTUP CHECK ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Blocks application startup until the database is fully reachable,
    then verifies and creates database tables before proceeding.
    """
    logger.info("🔍 Checking database connection status...")
    retries = 10
    while retries > 0:
        try:
            # Attempt a basic low-level connection test handshake
            with engine.connect() as connection:
                logger.info("✅ Database is reachable and online!")
                break
        except OperationalError:
            retries -= 1
            logger.warning(f"⏳ Database not ready yet. Retrying... ({retries} retries left)")
            time.sleep(2)
            
    if retries == 0:
        logger.error("❌ Fatal: Could not connect to the database. Exiting application setup.")
        raise RuntimeError("Database connection timed out.")

    # Automatically create missing tables (prediction_logs, etc.) based on your SQLAlchemy models
    logger.info("🛠️ Synchronizing database schema models...")
    Base.metadata.create_all(bind=engine)
    logger.info("🚀 Database schema up to date. Handing control to FastAPI.")
    
    yield  # The application runs and services requests here
    
    logger.info("🔌 Shutting down serving engine assets.")

# Pass the lifespan context manager into your FastAPI initialization
app = FastAPI(title="Production MLOps Serving Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, swap ["*"] for ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD ML MODEL ---
try:
    model = mlflow.pyfunc.load_model("models:/production_model/latest")
    logger.info('Latest model found and loaded successfully.')
except Exception as e:
    model = None 
    logger.error(f'No model found in registry: {e}')


@app.post("/predict")
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    features_dict = request.features
    
    # Run fallback / model prediction logic
    prediction = 1 
    if model is not None:
        try:
            # Simple conversion if needed for your specific model signature
            df_input = pd.DataFrame([features_dict])
            prediction = model.predict(df_input)[0]
        except Exception as e:
            logger.error(f"Prediction inference failed, defaulting: {e}")

    record = PredictionLog(
        features=features_dict,
        predicted_output=int(prediction)
    )
    db.add(record)
    db.commit()
    
    return {"prediction": int(prediction)}

@app.post("/monitor/check-drift")
async def check_drift_endpoint(background_tasks: BackgroundTasks):
    reference_df = pd.read_csv("ml/reference.csv")
    current_df = get_all_production_data()
    
    if current_df.empty:
        return {"status": "Insufficient Data", "action": "Log more predictions first."}
        
    drift_detected = check_data_drift(reference_df, current_df)
    
    if drift_detected:
        trigger_auto_retraining.delay()
        return {"status": "Drift Detected", "action": "Retraining triggered via Celery"}
        
    return {"status": "No Drift Detected", "action": "None"}

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    # 🛠️ THE FIX: Accept the handshake explicitly. 
    # If your FastAPI setup uses strict origin checking, you can accept it like this:
    await websocket.accept()
    
    logger.info("🔌 Frontend WebSocket client connected successfully!")
    try:
        while True:
            # Keep the connection open and listen/send data
            data = await websocket.receive_text()
            # (Your existing metric broadcast loop code goes here...)
            
    except WebSocketDisconnect:
        logger.info("🔌 Frontend WebSocket client disconnected.")