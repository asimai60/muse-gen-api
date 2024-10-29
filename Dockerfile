# Use slim version for smaller image size
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and credentials
COPY muse-gen-midi-files-keys.json /app/muse-gen-midi-files-keys.json
COPY . .

# Expose port
EXPOSE $PORT

# Run the application
CMD ["uvicorn", "vae_app:app", "--host", "0.0.0.0", "--port", "8080"]
