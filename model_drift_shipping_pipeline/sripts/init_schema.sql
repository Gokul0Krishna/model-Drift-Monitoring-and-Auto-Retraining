CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS shipping_records (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    warehouse_block VARCHAR(10) NOT NULL,
    mode_of_shipment VARCHAR(20) NOT NULL,
    customer_care_calls INT NOT NULL,
    customer_rating INT NOT NULL,
    cost_of_the_product NUMERIC(10, 2) NOT NULL,
    prior_purchases INT NOT NULL,
    product_importance VARCHAR(20) NOT NULL,
    gender VARCHAR(10) NOT NULL,
    discount_offered NUMERIC(10, 2) NOT NULL,
    weight_in_gms NUMERIC(10, 2) NOT NULL,
    ground_truth_reached_on_time INT NULL
);
CREATE INDEX IF NOT EXISTS idx_shipping_records_timestamp ON shipping_records(timestamp);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    shipping_record_id BIGINT UNIQUE NOT NULL REFERENCES shipping_records(id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL,
    predicted_late_probability FLOAT NOT NULL,
    predicted_class INT NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    eval_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reference_window_start TIMESTAMP NOT NULL,
    reference_window_end TIMESTAMP NOT NULL,
    current_window_start TIMESTAMP NOT NULL,
    current_window_end TIMESTAMP NOT NULL,
    feature_drift_scores JSONB NOT NULL,
    drift_detected BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(30) NOT NULL CHECK (status IN ('active_champion', 'evaluating_challenger', 'retired')),
    business_cost_score NUMERIC(15, 2) NOT NULL,
    metrics_metadata JSONB NOT NULL
);