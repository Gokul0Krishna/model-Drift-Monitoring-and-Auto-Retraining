import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import logging
from sklearn.compose import ColumnTransformer

logger = logging.getLogger("model_drift_shipping_pipeline.ml.pipeline")
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
   

def process_raw_data(action:str='train'):
    '''
    load data based on action(train/test) and split into x and y
    '''
    if action=='train':
        try:
            df = pd.read_csv('model_drift_shipping_pipeline/data/train_set.csv')
            logger.info('DATASET FOUND AND LOADED')
        except Exception as e:
            logger.error('EXCEPTION RAISED',e)
    elif action=='test':
        try:
            df = pd.read_csv('model_drift_shipping_pipeline/data/test_set.csv')
            logger.info('DATASET FOUND AND LOADED')
        except Exception as e:
            logger.error('EXCEPTION RAISED',e)
    else:
        logger.error('INVALID ACTION')
        raise ValueError("Invalid action")
    try:    
        procesed_df = _process_pipeline(df)
        logger.info('THE DATA HAS BEEN PROCESSED SUCCESSFULLY')
    except Exception as e:
        logger.error(f'AN EXCEPTION RAISED IN THE PREPROCESSING {e}',exc_info=True)
        raise ValueError("AN EXCEPTION RAISED IN THE PREPROCESSING")
    
    if action == 'train':
        X_train,X_test,y_train,y_test=train_test_split(procesed_df,df['remainder__Reached.on.Time_Y.N'],test_size=0.2)
        return X_train,X_test,y_train,y_test
    else:
        X = procesed_df.drop('remainder__Reached.on.Time_Y.N',axis=1)
        y = procesed_df['remainder__Reached.on.Time_Y.N']
        return X,y

if __name__ == '__main__':
    c1,c2 =process_raw_data(action='test')     
    print(c1[:5])
    print(c2[:5])