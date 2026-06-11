import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import logging
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')


logger = logging.getLogger(__name__)
categorical_features = ['Warehouse_block', 'Mode_of_Shipment','Product_importance', 'Gender']
        
def _process_pipeline(df:pd.DataFrame):
    '''
    converts categorical data into numeric using onehot encoding
    '''
    categorical_transformer = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
        ],
        verbose=True
    )

    processor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough', 
        verbose=True
    )

    full_pipeline = Pipeline(
        steps=[
            ('preprocessor', processor)
        ],
        verbose=True
    )

    preprocessed_data = full_pipeline.fit_transform(df)
    new_column_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
    processed_df = pd.DataFrame(preprocessed_data, columns=new_column_names, index=df.index)
    return processed_df
   

def process_raw_data(df=pd.DataFrame() ,action:str='train'):
    '''
    load data based on action(train/test) and split into x and y
    '''
    if action == 'train':
        try:
            df = pd.read_csv('model_drift_shipping_pipeline/data/train_set.csv')
            logger.info('DATASET FOUND AND LOADED')
        except Exception as e:
            logger.error('EXCEPTION RAISED',e)
    try:    
        procesed_df = _process_pipeline(df)
        logger.info('THE DATA HAS BEEN PROCESSED SUCCESSFULLY')
    except Exception as e:
        logger.error(f'AN EXCEPTION RAISED IN THE PREPROCESSING {e}',exc_info=True)
        raise ValueError("AN EXCEPTION RAISED IN THE PREPROCESSING")

    X = procesed_df.drop('remainder__Reached.on.Time_Y.N',axis=1)
    y = procesed_df['remainder__Reached.on.Time_Y.N']
    if action == 'train':
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
        return X_train,X_test,y_train,y_test
    else:
        return X,y

def slidding_windows_process(start_id:int,end_id:int, historical_lookback_limit=5000):
    try:
        engine = create_engine(DATABASE_URL)
        logger.info('DATABASE SUCCESFULLY CONNECTED')
    except Exception as e:
        logger.error(f'DATABASE CONNECTION ERROR: {e}')
        return    
    drift_query = f"""
        SELECT * FROM shipping_records 
        WHERE id BETWEEN {start_id} AND {end_id}
    """
    historical_query = f"""
        SELECT * FROM shipping_records 
        WHERE id < {start_id}
        ORDER BY id DESC
        LIMIT {historical_lookback_limit}
    """
    try:
        df_drift = pd.read_sql(drift_query, con=engine)
        logger.info('SUCCESSFULLY RETIVED DRIFT DATA')
        df_historical = pd.read_sql(historical_query, con=engine)
        logger.info('SUCCESSFULLY RETIVED HISTORICAL DATA')
    except Exception as e:
        logger.error(f'DATABASE QUERY ERROR: {e}')
        return
            
    combined_training_df = pd.concat([df_drift, df_historical], ignore_index=True)
    try:    
        procesed_df = _process_pipeline(combined_training_df)
        logger.info('THE DATA HAS BEEN PROCESSED SUCCESSFULLY')
    except Exception as e:
        logger.error(f'AN EXCEPTION RAISED IN THE PREPROCESSING {e}',exc_info=True)
        raise ValueError("AN EXCEPTION RAISED IN THE PREPROCESSING")

    X = procesed_df.drop('remainder__Reached.on.Time_Y.N',axis=1)
    y = procesed_df['remainder__Reached.on.Time_Y.N']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    return X_train,X_test,y_train,y_test


if __name__ == '__main__':
    c1,c2 =process_raw_data(action='test')     
    print(c1[:5])
    print(c2[:5])