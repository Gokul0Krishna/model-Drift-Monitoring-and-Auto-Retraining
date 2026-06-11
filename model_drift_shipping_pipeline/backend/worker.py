import shutil
from math import exp
import os
import pandas as pd
from celery import Celery
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
from pathlib import Path
import shutil
from mlflow.tracking import MlflowClient

from ml.monitor import run_check
from ml.train import train_and_save_challenger_model

load_dotenv()
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv('REDIS_URL')
DATABASE_URL = os.getenv('DATABASE_URL')
METRICS_DIR = BASE_DIR / "MLFLOW_TRACKING_URI/"
BASE_DIR = Path(__file__).resolve().parent.parent

CHAMPION_MODEL_PATH = BASE_DIR / "models/champion/shipping_rf_champion_model.pkl"
CHAMPION_MODEL = 'shipping_rf_champion_model'

CHALLENGER_MODEL_PATH = BASE_DIR / "models/challenger/shipping_rf_challenger_model.pkl"
CHALLENGER_MODEL = 'shipping_rf_challenger_model'

celery_app = Celery('tasks', broker=REDIS_URL)

@celery_app.task
def trigger_analysis(start_id, end_id):
    logger.info('STARTING DRIFT ANALYSIS')
    logger.info('STARTING DATA INGESTION')
    try:
        engine = create_engine(DATABASE_URL)
    except Exception as e:
        logger.error(f'DATABASE CONNECTION ERROR: {e}')
        return
    query = f"""
        SELECT warehouse_block, mode_of_shipment, customer_care_calls, 
               customer_rating, cost_of_the_product, prior_purchases, 
               product_importance, gender, discount_offered, weight_in_gms 
        FROM shipping_records 
        WHERE id BETWEEN {start_id} AND {end_id}
    """
    try:
        df = production_batch_df = pd.read_sql(query, con=engine)
    except Exception as e:
        logger.error(f'DATA INGESTION ERROR: {e}')
        engine.dispose()
        return
    engine.dispose()
    logger.info('DATA INGESTION SUCCESSFULL')
    logger.info('STARTING DRIFT CHECK')
    drift_detected = run_check(production_batch_df)
    logger.info(f'DRIFT DETECTED: {drift_detected}')


    return drift_detected

@celery_app.task()
def cost_eval(start_id,end_id):
    try:
        val_acc = train_and_save_challenger_model(start_id,end_id)
        logger.info(f'CHALLENGER MODEL TRAINING DONE, VALIDATION COST: {val_acc}')
    except Exception as e:
        logger.error(f'CHALLENGER MODEL TRAINING ERROR: {e}')
        return
    client = MlflowClient()
    champ_experiment = client.get_experiment_by_name(CHAMPION_MODEL)
    champ_runs = client.search_runs(experiment_ids=[champ_experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not champ_runs:
        logger.error("No runs found in the experiment")
        return
    best_run = champ_runs[0]
    champ_run_cost = best_run.data.metrics.get('validation_loss')

    chall_experiment = client.get_experiment_by_name(CHALLENGER_MODEL)
    chall_runs = client.search_runs(experiment_ids=[chall_experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not chall_runs:
        logger.error("No runs found in the experiment")
        return
    chall_run = chall_runs[0]
    chall_run_cost = chall_run.data.metrics.get('validation_loss')
    

    if chall_run_cost<champ_run_cost:
            shutil.copy2(CHALLENGER_MODEL_PATH, CHAMPION_MODEL_PATH)
            os.remove(CHALLENGER_MODEL_PATH)
            return True
    else:
        os.remove(CHALLENGER_MODEL_PATH)
    
    return False
    

   

    
