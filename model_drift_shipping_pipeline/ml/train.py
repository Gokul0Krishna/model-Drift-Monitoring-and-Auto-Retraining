import joblib
from sklearn.ensemble import RandomForestClassifier
from pipeline import process_raw_data
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("model_drift_shipping_pipeline.ml.train")

def train_and_save_model(model_name:str='model',model_path:str = MODEL_DIR):
    logger.info('TRAINING THE MODEL')
    try:
        X_train,X_test,y_train,y_test = process_raw_data(action='train')
        logger.info('PROCESSED DATA LOADED')
        model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
        model.fit(X_train,y_train)
        joblib.dump(model,MODEL_DIR/f'{model_name}.pkl')
        logger.info('MODEL TRAINED AND SAVED')
    except Exception as e:
        logger.info('EXCEPTION RAISED',e)
        raise e

if __name__ == '__main__':
    train_and_save_model()
    