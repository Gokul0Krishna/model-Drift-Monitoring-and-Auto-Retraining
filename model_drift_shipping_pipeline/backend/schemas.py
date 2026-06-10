from pydantic import BaseModel
from typing import List, Optional

class ShippingRecordItem(BaseModel):
    warehouse_block: str
    mode_of_shipment: str
    customer_care_calls: int
    customer_rating: int
    cost_of_the_product: float
    prior_purchases: int
    product_importance: str
    gender: str
    discount_offered: float
    weight_in_gms: float

class IngestionPayload(BaseModel):
    records: List[ShippingRecordItem]