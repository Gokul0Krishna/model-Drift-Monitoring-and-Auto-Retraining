import logging
import pandas as pd
from scipy.stats import ks_2samp
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"/'train_set.csv'

def run_check(production_df:pd.DataFrame):
    if not os.path.exists(DATA_DIR):
        logger.error("DATA NOT FOUND")
        return {
            "message": "Production data not found",
            "status": "error"
        }
    logger.info('DATA FOUND')
    train_df = pd.read_csv(DATA_DIR)
    numerical_features = ['cost_of_the_product', 'discount_offered', 'weight_in_gms', 'customer_care_calls']
    drift_threshold = 0.05
    drift_detected = False
    
    for feature in numerical_features:
        if feature in production_df.columns and feature in train_df.columns:
            stat, p_value = ks_2samp(production_df[feature], train_df[feature])
            if p_value < drift_threshold:
                logger.warning(f"DRIFT DETECTED in feature: {feature}")
                drift_detected = True
        else:
            logger.error(f"FEATURE NOT FOUND: {feature}")
    
    return drift_detected
