from sqlalchemy import Column, Integer, Float, Boolean, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
import datetime
from app.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    features = Column(JSONB, nullable=False)
    predicted_output = Column(Integer, nullable=False)
    ground_truth = Column(Integer, nullable=True)
    predicted_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    check_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    dataset_version = Column(String(50), nullable=False)
    drift_score = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    report_json_path = Column(String(255), nullable=True)