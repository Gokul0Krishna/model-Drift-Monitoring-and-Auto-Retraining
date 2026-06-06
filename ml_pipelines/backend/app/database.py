import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings
import logging

logger = logging.getLogger(__name__)
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to inject DB sessions into FastAPI routes cleanly
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_all_production_data() -> pd.DataFrame:
    """
    Queries the prediction logs directly into a Pandas DataFrame 
    unwrapping JSONB columns for processing in Evidently AI.
    """
    query = "SELECT features, predicted_output FROM prediction_logs ORDER BY predicted_at DESC LIMIT 5000;"
    df_raw = pd.read_sql_query(query, con=engine)
    
    if df_raw.empty:
        return pd.DataFrame()
        
    # Unpack the list of feature dictionaries into separate columns
    features_df = pd.json_normalize(df_raw['features'])
    features_df['target'] = df_raw['predicted_output']
    
    return features_df