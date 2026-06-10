FROM python:3.12-slim

# Set working directory to the pipeline root
WORKDIR /app

# Install system dependencies required by psycopg2-binary, scipy, scikit-learn, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first for better Docker layer caching
COPY pyproject.toml ./

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Copy all source packages and data
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY data/ ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Default port for uvicorn
EXPOSE 8000
