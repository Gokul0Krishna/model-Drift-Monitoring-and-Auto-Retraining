from celery import Celery
import os
import pandas as pd
from ml.pipeline import train_and_register_model
from app.database import get_all_production_data
import logging

logger = logging.getLogger(__name__) 
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task
def trigger_auto_retraining():
    logger.info('RETRAINING STARTED')
    reference_df = pd.read_csv("ml/reference.csv")
    current_df = get_all_production_data() 
    
    if len(current_df) < 500:
        logger.error("Not enough new data samples to retrain.")
        return "Not enough new data samples to retrain."
        
    combined_df = pd.concat([reference_df, current_df]).drop_duplicates()
    
    train_df = combined_df.sample(frac=0.8, random_state=42)
    val_df = combined_df.drop(train_df.index)
    
    model_uri, new_auc, pre, rec = train_and_register_model(train_df, val_df)
    
    logger.info('RETRAINING COMPLETED')
    logger.info(f"Success! New model registered at {model_uri} with AUC: {new_auc}, precision: {pre}, recall: {rec}")
    return f"Success! New model registered at {model_uri} with AUC: {new_auc}, precision: {pre}, recall: {rec}"