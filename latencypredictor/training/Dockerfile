FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# FIX: GHSA-4xh5-x5gv-qwph 
RUN pip install --upgrade pip==25.3

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove build deps (resolves: CVE-2018-12928)
# RUN apt-get purge -y --auto-remove \
#     gcc g++ libc6-dev linux-libc-dev \
#   && rm -rf /var/lib/apt/lists/*

# Copy shared common code
COPY common/ ./common/

# Copy training server code
COPY training/ ./training/

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# Use Python module syntax to run from training package
CMD ["python", "-m", "uvicorn", "training.training_server:app", "--host", "0.0.0.0", "--port", "8000"]
