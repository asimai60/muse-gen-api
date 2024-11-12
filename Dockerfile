# Build stage
FROM python:3.9-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only the necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY muse-gen-midi-files-keys.json /app/muse-gen-midi-files-keys.json
COPY . .

# Expose port
EXPOSE $PORT

# Run the application
CMD ["uvicorn", "vae_app:app", "--host", "0.0.0.0", "--port", "8080"]
