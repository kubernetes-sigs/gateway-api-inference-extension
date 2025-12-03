FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# FIX: GHSA-4xh5-x5gv-qwph 
RUN pip install --upgrade pip==25.3

# Install Python dependencies for testing (minimal set)
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    requests \
    httpx

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional test dependencies
RUN pip install --no-cache-dir httpx

# Copy shared common code (test may need to inspect bundle constants)
COPY common/ ./common/

# Copy test files
COPY test/ ./test/

# Create test results directory
RUN mkdir -p /test-results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Use entrypoint to handle warmup logic
ENTRYPOINT ["/app/test/test_entrypoint.sh"]

# Default command runs the specific test
CMD ["pytest", "-v", "-s", "test/test_dual_server_client.py"]
