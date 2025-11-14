FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CPU support (for containers without GPU)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/audio /app/models

# Set Python path to include shared libraries
ENV PYTHONPATH="/app:/app/shared-libs:$PYTHONPATH"

# Default command (can be overridden in docker-compose)
CMD ["celery", "worker", "-A", "src.celery_app", "--loglevel=info", "--concurrency=2"]