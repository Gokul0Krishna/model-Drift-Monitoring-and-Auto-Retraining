import requests
import time
import random

API_URL = "http://localhost:8000/predict"

def send_prediction_request(features):
    try:
        response = requests.post(API_URL, json={"features": features})
        return response.json()
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None

print("🚀 Starting MLOps Pipeline End-to-End Test...")

# --- PHASE 1: SIMULATE NORMAL TRAFFIC ---
print("\n📊 Step 1: Sending 50 normal production requests...")
for i in range(50):
    # Simulating standard features (e.g., normal credit scores, income, ages)
    normal_features = {
        "credit_score": random.randint(600, 850),
        "income": random.randint(30000, 120000),
        "age": random.randint(22, 65)
    }
    send_prediction_request(normal_features)
    time.sleep(0.05) # Fast streaming
print("✅ Normal traffic simulation complete. Data logged to PostgreSQL.")

# --- PHASE 2: INTRODUCE SEVERE FEATURE DRIFT ---
print("\n⚠️ Step 2: Injecting severe Feature Drift to trigger auto-retraining...")
for i in range(50):
    # Artificially crashing credit scores and spiking ages to break the model's distributions
    drifted_features = {
        "credit_score": random.randint(300, 450),  # Massively dropped
        "income": random.randint(15000, 25000),    # Drastically lowered
        "age": random.randint(80, 100)             # Heavily aged demographic
    }
    send_prediction_request(drifted_features)
    time.sleep(0.05)
print("✅ Drifted traffic injected successfully.")

# --- PHASE 3: TRIGGER THE MONITORING CHECK ---
print("\n🔍 Step 3: Triggering drift analysis endpoint...")
try:
    # Assuming your pipeline has an explicit endpoint to run the check
    # Alternatively, if your pipeline runs on a Celery beat schedule, you can just wait.
    monitoring_response = requests.post("http://localhost:8000/monitor/check-drift")
    print(f"Server Response: {monitoring_response.json()}")
except Exception:
    print("Sent monitoring trigger request. Check your Docker logs to watch the magic happen!")