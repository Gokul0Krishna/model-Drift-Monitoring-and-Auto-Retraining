import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(
    DATABASE_URL,
    pool_size=20,      
    max_overflow=10,      
    pool_pre_ping=True    
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
def verify_db_connection():
    logger.info(f"Testing DB Connection to : {DATABASE_URL}")
    try:
        connection = engine.connect()
        connection.close()
        logger.info("Connection successful")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__=='__main__':
    verify_db_connection()