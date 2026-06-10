from sqlalchemy import Column, BigInteger, String, Integer, Numeric, DateTime, func
from .database import Base

class ShippingRecordModel(Base):
    __tablename__ = "shipping_records"

    id = Column(BigInteger, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    warehouse_block = Column(String(10), nullable=False)
    mode_of_shipment = Column(String(20), nullable=False)
    customer_care_calls = Column(Integer, nullable=False)
    customer_rating = Column(Integer, nullable=False)
    cost_of_the_product = Column(Numeric(10, 2), nullable=False)
    prior_purchases = Column(Integer, nullable=False)
    product_importance = Column(String(20), nullable=False)
    gender = Column(String(10), nullable=False)
    discount_offered = Column(Numeric(10, 2), nullable=False)
    weight_in_gms = Column(Numeric(10, 2), nullable=False)
    ground_truth_reached_on_time = Column(Integer, nullable=True)