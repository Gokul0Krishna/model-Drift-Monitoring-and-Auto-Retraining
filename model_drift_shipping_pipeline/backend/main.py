from fastapi import FastAPI, BackgroundTasks, Depends
import mlflow.pyfunc
import pandas as pd
import logging
import time
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect

from .database import engine, Base, get_db

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

try:
    model = mlflow.pyfunc.load_model("models:/production_model/latest")
    logger.info('Latest model found and loaded successfully.')
except Exception as e:
    model = None 
    logger.error(f'No model found in registry: {e}')