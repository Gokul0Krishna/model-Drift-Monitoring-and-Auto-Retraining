from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
import pandas as pd
import logging
import time
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
from pathlib import Path
from sqlalchemy.orm import Session
from typing import List



from .database import engine, Base, get_db
from .worker import trigger_analysis,eval
from .schemas import IngestionPayload
from .model import ShippingRecordModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_LOC = BASE_DIR / 'models/shipping_rf_model.pkl'
TEST_LOC = BASE_DIR / 'data/test_set.csv'
# --- LIFESPAN STARTUP CHECK ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Blocks application startup until the database is fully reachable,
    then verifies and creates database tables before proceeding.
    """
    logger.info("Checking database connection status...")
    retries = 5
    while retries > 0:
        try:
            with engine.connect() as connection:
                logger.info("Database is reachable and online!")
                break
        except OperationalError:
            retries -= 1
            logger.warning(f"Database not ready yet. Retrying... ({retries} retries left)")
            time.sleep(2)
            
    if retries == 0:
        logger.error(" Fatal: Could not connect to the database. Exiting application setup.")
        raise RuntimeError("Database connection timed out.")

    logger.info("Synchronizing database schema models...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema up to date. Handing control to FastAPI.")
    
    yield  
    
    logger.info("Shutting down serving engine assets.")

app = FastAPI(title="Production MLOps Serving Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, swap ["*"] for ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ingest")
def ingest_data(payload: IngestionPayload, db: Session = Depends(get_db)):
    if not payload.records:
        logger.warning("Payload record collection is empty.")
        raise HTTPException(status_code=400, detail="Payload record collection is empty.")
    logger.info(f'PAYLOAD RECIVED -- {len(payload.records)} -- records')
    try:
        inserted_objects = []
        for record in payload.records:
            db_record = ShippingRecordModel(**record.model_dump())
            inserted_objects.append(db_record)
        
        db.bulk_save_objects(inserted_objects, return_defaults=True)
        db.commit()
        logger.info(f'PUSHED PAYLOAD TO THE DATABASE -- {len(inserted_objects)} -- records')
        start_id = inserted_objects[0].id
        end_id = inserted_objects[-1].id
        logger.info(f"Successfully stored {len(inserted_objects)} records. ID Range: {start_id} - {end_id}")

        drift_analyisis = trigger_analysis.delay(start_id, end_id)
        

        return {
            'analysis_id': drift_analyisis.id,
            "records_ingested": len(inserted_objects),
            "batch_range": {"start_id": start_id, "end_id": end_id}
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Failed handling ingestion storage stream: {e}")
        raise HTTPException(status_code=500, detail="Database write failure.")


@app.post('/retrain_model')
def retrain_model(start_id:int,end_id:int):
    eval(start_id,end_id)
        
    
        