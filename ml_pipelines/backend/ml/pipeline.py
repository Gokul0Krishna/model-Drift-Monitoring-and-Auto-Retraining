import mlflow
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,precision_score,recall_score
import pandas as pd

def train_and_register_model(train_df, val_df, model_name="production_model"):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Model_Retraining_Pipeline")
    
    X_train, y_train = train_df.drop(columns=['target']), train_df['target']
    X_val, y_val = val_df.drop(columns=['target']), val_df['target']
    
    with mlflow.start_run() as run:
        params = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 100, "objective": "binary:logistic"}
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        # pre = precision_score(y_val,preds)
        # rec = recall_score(y_val,preds)
        # mlflow.log_metric("auc", auc)
        mlflow.log_metrics({"auc": auc})

        input_example = X_train.iloc[[0]]
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=input_example
        )
        
        return model_info.model_uri, auc