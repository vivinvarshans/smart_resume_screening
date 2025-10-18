# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY apps/backend/requirements.txt ./apps/backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r apps/backend/requirements.txt

# Copy backend application
COPY apps/backend ./apps/backend

# Make start script executable
RUN chmod +x ./apps/backend/start.sh

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Set working directory to backend
WORKDIR /app/apps/backend

# Start command
CMD ["./start.sh"]
