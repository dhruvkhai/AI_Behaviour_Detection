# Use Python 3.12.9 slim as base image for the API
FROM python:3.12.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# Note: For GPU support inside Docker (e.g., for model training), 
# use a CUDA base image like nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
