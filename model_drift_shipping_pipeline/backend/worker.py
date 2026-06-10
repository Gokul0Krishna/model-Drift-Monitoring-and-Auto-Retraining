import os
import pandas as pd
from celery import Celery
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
from ..ml.monitor import run_drift_check

load_dotenv()
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv('REDIS_URL')
DATABASE_URL = os.getenv('DATABASE_URL')

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
    drift_detected = run_drift_check(production_batch_df)
    logger.info(f'DRIFT DETECTED: {drift_detected}')


    return drift_detected
