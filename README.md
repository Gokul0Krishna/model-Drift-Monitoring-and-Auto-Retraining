# 📦 Model Drift Monitoring & Auto-Retraining Pipeline

An end-to-end **MLOps pipeline** for an e-commerce shipping delay prediction system. It automatically ingests production data, detects **data drift** using statistical tests, retrains a **challenger model** on fresh data (via a sliding-window strategy), and **promotes it to champion** only when it outperforms the current model on a custom business-cost metric — all orchestrated through a FastAPI backend, Celery workers, and MLflow experiment tracking.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Configure Environment Variables](#2-configure-environment-variables)
  - [3. Start the Infrastructure (Docker Compose)](#3-start-the-infrastructure-docker-compose)
  - [4. Local Development (Without Docker)](#4-local-development-without-docker)
- [Usage](#usage)
  - [Training the Champion Model](#training-the-champion-model)
  - [Ingesting Data & Triggering Drift Analysis](#ingesting-data--triggering-drift-analysis)
  - [Retraining & Challenger Evaluation](#retraining--challenger-evaluation)
  - [Running the Test Script](#running-the-test-script)
- [API Reference](#api-reference)
- [How Drift Detection Works](#how-drift-detection-works)
- [Champion vs Challenger Strategy](#champion-vs-challenger-strategy)
- [CI/CD Pipeline](#cicd-pipeline)
- [Database Schema](#database-schema)
- [License](#license)

---

## Architecture Overview

```
                         ┌─────────────────┐
                         │   Client / Test  │
                         │    Script        │
                         └────────┬────────┘
                                  │  POST /ingest
                                  ▼
                         ┌─────────────────┐
                         │   FastAPI        │
                         │   Backend API    │
                         │   (port 8000)    │
                         └──┬──────────┬───┘
                            │          │
                  Store records   Enqueue Celery task
                            │          │
                  ┌─────────▼──┐  ┌────▼──────────┐
                  │ PostgreSQL │  │  Redis Broker  │
                  │  (port     │  │  (port 6379)   │
                  │   5432)    │  └────┬───────────┘
                  └────────────┘       │
                                       ▼
                              ┌──────────────────┐
                              │  Celery Worker    │
                              │  ┌──────────────┐ │
                              │  │ Drift Check  │ │  ──▶  KS-Test on numerical features
                              │  │ (Evidently/  │ │
                              │  │  SciPy)      │ │
                              │  └──────┬───────┘ │
                              │         │ if drift │
                              │  ┌──────▼───────┐ │
                              │  │  Retrain     │ │  ──▶  Sliding-window challenger model
                              │  │  Challenger  │ │
                              │  └──────┬───────┘ │
                              │         │         │
                              │  ┌──────▼───────┐ │
                              │  │  Cost Eval   │ │  ──▶  Promote if challenger cost < champion cost
                              │  │  & Promote   │ │
                              │  └──────────────┘ │
                              └────────┬──────────┘
                                       │ Log metrics
                                       ▼
                              ┌──────────────────┐
                              │  MLflow Tracking  │
                              │  Server           │
                              │  (port 5000)      │
                              └──────────────────┘
```

---

## Tech Stack

| Layer               | Technology                                         |
| ------------------- | -------------------------------------------------- |
| **API Framework**   | FastAPI + Uvicorn                                  |
| **Task Queue**      | Celery + Redis                                     |
| **Database**        | PostgreSQL 15                                      |
| **ML Framework**    | scikit-learn (RandomForestClassifier), XGBoost      |
| **Experiment Tracking** | MLflow 2.11                                    |
| **Drift Detection** | SciPy (Kolmogorov-Smirnov test)                    |
| **Data Processing** | pandas, scikit-learn Pipelines (OneHotEncoder)     |
| **Containerisation**| Docker & Docker Compose                            |
| **CI/CD**           | GitHub Actions                                     |
| **Language**        | Python 3.12                                        |

---

## Project Structure

```
model-Drift-Monitoring-and-Auto-Retraining/
├── docker-compose.yml              # Orchestrates all services
├── .env.template                   # Template for environment variables
├── .gitignore
├── README.md
│
└── model_drift_shipping_pipeline/  # Main application package
    ├── Dockerfile                  # Docker image for backend & worker
    ├── pyproject.toml              # Python project metadata & dependencies
    │
    ├── backend/                    # FastAPI application
    │   ├── main.py                 # API endpoints (/ingest, /retrain_model)
    │   ├── worker.py               # Celery tasks (drift analysis, cost evaluation)
    │   ├── database.py             # SQLAlchemy engine & session management
    │   ├── model.py                # ORM model (ShippingRecordModel)
    │   └── schemas.py              # Pydantic request/response schemas
    │
    ├── ml/                         # Machine learning modules
    │   ├── train.py                # Champion & challenger model training
    │   ├── pipeline.py             # Data preprocessing & sliding-window logic
    │   └── monitor.py              # Drift detection (KS-test)
    │
    ├── data/                       # Training & test datasets
    │   ├── train_set.csv
    │   ├── test_set.csv
    │   └── transform.ipynb         # Data exploration / transformation notebook
    │
    ├── models/                     # Saved model binaries (.pkl)
    │   ├── champion/
    │   └── challenger/
    │
    ├── scripts/
    │   └── init_schema.sql         # PostgreSQL schema initialisation
    │
    ├── test/
    │   └── test_run.py             # End-to-end integration test script
    │
    └── .github/
        └── workflows/
            └── ci-cd.yml           # GitHub Actions CI/CD pipeline
```

---

## Prerequisites

Make sure the following are installed on your machine:

- **Python** 3.10+ → [Download](https://www.python.org/downloads/)
- **Docker** & **Docker Compose** → [Download](https://docs.docker.com/get-docker/)
- **Git** → [Download](https://git-scm.com/downloads)

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Gokul0Krishna/model-Drift-Monitoring-and-Auto-Retraining.git
cd model-Drift-Monitoring-and-Auto-Retraining
```

### 2. Configure Environment Variables

Copy the template and fill in your values:

```bash
cp .env.template .env
```

Edit `.env` with the following (these defaults work with the Docker Compose setup):

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres_password
POSTGRES_DB=mlops_db
DATABASE_URL=postgresql://postgres:postgres_password@db:5432/mlops_db

REDIS_URL=redis://redis:6379/0
MLFLOW_TRACKING_URI=http://mlflow:5000
DATASET_PATH=data/train_set.csv
```

> **Note:** When running outside Docker (locally), replace hostnames (`db`, `redis`, `mlflow`) with `localhost`.

### 3. Start the Infrastructure (Docker Compose)

This single command spins up **all 5 services**: PostgreSQL, Redis, MLflow, FastAPI backend, and the Celery worker.

```bash
docker-compose up --build
```

Once everything is running, you can access:

| Service              | URL                          |
| -------------------- | ---------------------------- |
| **FastAPI Docs**     | http://localhost:8000/docs    |
| **MLflow UI**        | http://localhost:5000         |
| **PostgreSQL**       | `localhost:5432`              |
| **Redis**            | `localhost:6379`              |

To stop all services:

```bash
docker-compose down
```

To also remove persisted data volumes:

```bash
docker-compose down -v
```

### 4. Local Development (Without Docker)

If you prefer running the app directly on your machine:

**a) Create and activate a virtual environment:**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**b) Install dependencies:**

```bash
cd model_drift_shipping_pipeline
pip install -e ".[dev]"
```

**c) Start required services (PostgreSQL, Redis, MLflow):**

You still need these services running. You can either install them natively or run just the infrastructure containers:

```bash
# From the project root (not inside model_drift_shipping_pipeline/)
docker-compose up db redis mlflow
```

**d) Update `.env` for local hostnames:**

```env
DATABASE_URL=postgresql://postgres:postgres_password@localhost:5432/mlops_db
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000
```

**e) Run the FastAPI server:**

```bash
cd model_drift_shipping_pipeline
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**f) Run the Celery worker (in a separate terminal):**

```bash
cd model_drift_shipping_pipeline
celery -A backend.worker.celery_app worker --loglevel=info
```

---

## Usage

### Training the Champion Model

To train the initial champion model from the training dataset:

```bash
cd model_drift_shipping_pipeline
python -m ml.train
```

This will:
1. Load and preprocess `data/train_set.csv`
2. Train a RandomForest classifier
3. Log parameters and metrics to MLflow
4. Save the model binary to `models/champion/`

### Ingesting Data & Triggering Drift Analysis

Send production data batches to the `/ingest` endpoint. The system will automatically:
1. Store records in PostgreSQL
2. Enqueue a Celery task for drift analysis
3. Run KS-tests on key numerical features against the training distribution

**Example with `curl`:**

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "warehouse_block": "A",
        "mode_of_shipment": "Ship",
        "customer_care_calls": 4,
        "customer_rating": 2,
        "cost_of_the_product": 177,
        "prior_purchases": 3,
        "product_importance": "Low",
        "gender": "M",
        "discount_offered": 44,
        "weight_in_gms": 1233
      }
    ]
  }'
```

### Retraining & Challenger Evaluation

When drift is detected, trigger retraining via:

```bash
curl -X POST "http://localhost:8000/retrain_model?start_id=1&end_id=1000"
```

This will:
1. Train a challenger model on a **sliding window** of recent + historical data
2. Compare the challenger's validation cost against the champion's
3. **Automatically promote** the challenger if it achieves a lower business cost

### Running the Test Script

The included test script sends the entire test dataset in batches and automatically triggers retraining when drift is detected:

```bash
python model_drift_shipping_pipeline/test/test_run.py
```

---

## API Reference

| Method | Endpoint           | Description                                                       |
| ------ | ------------------ | ----------------------------------------------------------------- |
| `POST` | `/ingest`          | Ingest a batch of shipping records; triggers async drift analysis |
| `POST` | `/retrain_model`   | Retrain challenger model and evaluate against champion            |
| `GET`  | `/docs`            | Interactive Swagger UI (auto-generated by FastAPI)                |
| `GET`  | `/redoc`           | Alternative API documentation                                    |

### `POST /ingest` — Request Body

```json
{
  "records": [
    {
      "warehouse_block": "string",
      "mode_of_shipment": "string",
      "customer_care_calls": 0,
      "customer_rating": 0,
      "cost_of_the_product": 0.0,
      "prior_purchases": 0,
      "product_importance": "string",
      "gender": "string",
      "discount_offered": 0.0,
      "weight_in_gms": 0.0
    }
  ]
}
```

### `POST /ingest` — Response

```json
{
  "analysis_id": "celery-task-id",
  "records_ingested": 100,
  "batch_range": {
    "start_id": 1,
    "end_id": 100
  }
}
```

---

## How Drift Detection Works

The pipeline uses the **Kolmogorov-Smirnov (KS) two-sample test** to detect distribution shifts between the training data and incoming production batches.

**Monitored features:**
- `cost_of_the_product`
- `discount_offered`
- `weight_in_gms`
- `customer_care_calls`

**Drift threshold:** `p-value < 0.05`

If any feature's p-value falls below the threshold, the system flags **drift detected** and triggers the retraining pipeline.

---

## Champion vs Challenger Strategy

The retraining system uses a **Champion–Challenger** pattern:

1. **Champion Model** — The currently deployed model serving predictions
2. **Challenger Model** — A newly trained model on recent + historical data (sliding window)

**Promotion logic:**
- Both models are evaluated using a **custom business-cost metric**:
  - **False Positive** (predicted late, arrived on-time): **$5** (unnecessary voucher/notification)
  - **False Negative** (predicted on-time, arrived late): **$50** (SLA penalty / support cost)
- The challenger **replaces** the champion only if its validation cost is **strictly lower**
- All training runs, parameters, and metrics are logged to **MLflow** for auditability

---

## CI/CD Pipeline

The project includes a **GitHub Actions** workflow (`.github/workflows/ci-cd.yml`) that runs on every push/PR to `main`:

| Stage               | Description                                              |
| ------------------- | -------------------------------------------------------- |
| **Test**            | Spins up a PostgreSQL service container and runs `pytest` |
| **Build Docker**    | Builds the backend and dashboard Docker images (no push)  |

> To enable pushing images to a registry (DockerHub, AWS ECR, etc.), add your registry secrets and set `push: true` in the workflow.

---

## Database Schema

The PostgreSQL database is initialised with four tables:

| Table               | Purpose                                                  |
| ------------------- | -------------------------------------------------------- |
| `shipping_records`  | Stores all ingested production shipping data             |
| `predictions`       | Stores model predictions per record                      |
| `drift_metrics`     | Logs drift evaluation results (feature scores, windows)  |
| `model_registry`    | Tracks model lifecycle (champion, challenger, retired)    |

---

## License

This project is open-source. Feel free to use, modify, and distribute.
