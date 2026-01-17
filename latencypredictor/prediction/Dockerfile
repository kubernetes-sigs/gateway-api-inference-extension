FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# FIX: GHSA-4xh5-x5gv-qwph 
RUN pip install --upgrade pip==25.3

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared common code
COPY common/ ./common/

# Copy prediction server code
COPY prediction/ ./prediction/

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application using uvicorn
# Single worker with high connection limit for async concurrency
CMD ["python", "-m", "uvicorn", "prediction.prediction_server:app", "--host", "0.0.0.0", "--port", "8001"]
