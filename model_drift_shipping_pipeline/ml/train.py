import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
from sklearn.ensemble import RandomForestClassifier
from ml.pipeline import process_raw_data
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
import pandas as pd

from backend.schemas import ShippingRecordItem 
from sqlalchemy import create_engine


os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
DATABASE_URL = os.getenv("DATABASE_URL")

BASE_DIR = Path(__file__).resolve().parent.parent

CHAMPION_MODEL_DIR = BASE_DIR / "models/champion/"
CHAMPION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHALLENGER_MODEL_DIR = BASE_DIR / "models/challenger/"
CHALLENGER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("MLFLOW_TRACKING_URI") 
mlflow.set_experiment("model-drift-shipping-pipeline")

def calculate_logistics_cost(y_true, y_pred) -> float:
    """
    Computes custom financial penalty thresholds based on e-commerce logistics outcomes:
    - False Positive (Predicted Late, arrived On-Time): Low Cost ($5 notification/voucher)
    - False Negative (Predicted On-Time, arrived Late): Critical Cost ($50 SLA penalty/support call)
    """
    total_cost = 0.0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 1:   # False Positive
            total_cost += 5.0
        elif true == 1 and pred == 0: # False Negative (Missed Delay - Catastrophic)
            total_cost += 50.0
    return total_cost

def train_and_save_champion_model(model_name: str = 'shipping_rf_champion_model'):
    '''
    Trains an RF model, logs metrics/params to MLflow, and registers the binary.
    '''
    logger.info('TRAINING THE MODEL')
    try:
        X_train, X_test, y_train, y_test = process_raw_data(action='train')
        logger.info('PROCESSED DATA LOADED')
        params = {
        'n_estimators' : 100,
        'max_depth' : 10,
        'random_state' : 42
        }
        model = RandomForestClassifier(**params)
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run Started: {run.info.run_id}")
            
            model.fit(X_train, y_train)
            
            train_acc = calculate_logistics_cost(X_train, y_train)
            val_acc = calculate_logistics_cost(X_test, y_test)

            mlflow.log_param("n_estimators", params['n_estimators'])
            mlflow.log_param("max_depth", params['max_depth'])
            mlflow.log_metric("train_cost", train_acc)
            mlflow.log_metric("val_cost", val_acc)
            
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model_artifacts",
                registered_model_name=model_name,
                serialization_format="skops"
            )
            joblib.dump(model, CHAMPION_MODEL_DIR / f'{model_name}.pkl')
            
            logger.info('MODEL TRAINED, LOGGED TO MLFLOW, AND SAVED LOCAL')
        
        return {"val_cost": val_acc}
            
    except Exception as e:
        logger.error(f'EXCEPTION RAISED: {e}', exc_info=True)
        raise e

def train_and_save_challenger_model(start_id: int, end_id: int, model_name: str = 'shipping_rf_challenger_model'):
    engine = create_engine(DATABASE_URL)
    prod_query = f"""
        SELECT warehouse_block, mode_of_shipment, customer_care_calls, 
               customer_rating, cost_of_the_product, prior_purchases, 
               product_importance, gender, discount_offered, weight_in_gms,
               ground_truth_reached_on_time
        FROM shipping_records 
        WHERE id BETWEEN {start_id} AND {end_id}
          AND ground_truth_reached_on_time IS NOT NULL
    """
    prod_df = pd.read_sql(prod_query, con=engine)
    if prod_df.empty:
        logger.error("NO DATA FOR TRAINING")
        return

    base_query = """
        SELECT warehouse_block, mode_of_shipment, customer_care_calls, 
               customer_rating, cost_of_the_product, prior_purchases, 
               product_importance, gender, discount_offered, weight_in_gms,
               ground_truth_reached_on_time
        FROM shipping_records
        WHERE id < 5000 
          AND ground_truth_reached_on_time IS NOT NULL
        LIMIT 3000
    """
    base_df = pd.read_sql(base_query, con=engine)
    if base_df.empty:
        logger.error("NO DATA FOR TRAINING")
        return

    training_df = pd.concat([base_df, prod_df], ignore_index=True)
    X_train, X_test, y_train, y_test = process_raw_data(df=training_df, action='train')
    logger.info('TRAINING THE MODEL')
    try:
        logger.info('PROCESSED DATA LOADED')
        params = {
        'n_estimators' : 100,
        'max_depth' : 10,
        'random_state' : 42
        }
        model = RandomForestClassifier(**params)
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run Started: {run.info.run_id}")
            
            model.fit(X_train, y_train)
            
            train_acc = calculate_logistics_cost(X_train, y_train)
            val_acc = calculate_logistics_cost(X_test, y_test)

            mlflow.log_param("n_estimators", params['n_estimators'])
            mlflow.log_param("max_depth", params['max_depth'])
            mlflow.log_metric("train_cost", train_acc)
            mlflow.log_metric("val_cost", val_acc)
            
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model_artifacts",
                registered_model_name=model_name,
                serialization_format="skops"
            )
            joblib.dump(model, CHALLENGER_MODEL_DIR / f'{model_name}.pkl')
            
            logger.info('MODEL TRAINED, LOGGED TO MLFLOW, AND SAVED LOCAL')
        return {"val_cost": val_acc}
            
    except Exception as e:
        logger.error(f'EXCEPTION RAISED: {e}', exc_info=True)
        raise e
    

    
if __name__ == '__main__':
    print(train_and_save_champion_model())