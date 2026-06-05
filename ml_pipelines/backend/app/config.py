import os

class Settings:
    PROJECT_NAME: str = "Automated MLOps Production Engine"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres_password@db:5432/mlops_db"
    )
    
    # Task Queue Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # MLOps Tracking Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    
    # Reference Data Path
    REFERENCE_DATA_PATH: str = os.getenv("REFERENCE_DATA_PATH", "ml/reference.csv")

settings = Settings()