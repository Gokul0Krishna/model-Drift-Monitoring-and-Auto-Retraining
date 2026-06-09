import joblib
from sklearn.ensemble import RandomForestClassifier
from pipeline import process_raw_data
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("model_drift_shipping_pipeline.ml.train")

mlflow.set_tracking_uri("MLFLOW_TRACKING_URI") 
mlflow.set_experiment("model-drift-shipping-pipeline")

def train_and_save_model(model_name: str = 'shipping_rf_model'):
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
            
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_test, y_test)

            mlflow.log_param("n_estimators", params['n_estimators'])
            mlflow.log_param("max_depth", params['max_depth'])
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", val_acc)
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifacts",
                registered_model_name=model_name 
            )
            joblib.dump(model, MODEL_DIR / f'{model_name}.pkl')
            
            logger.info('MODEL TRAINED, LOGGED TO MLFLOW, AND SAVED LOCAL')
            
    except Exception as e:
        logger.error(f'EXCEPTION RAISED: {e}', exc_info=True)
        raise e

if __name__ == '__main__':
    train_and_save_model()