# Multi-stage build: Build both TreeLite and tl2cgen from source
# Use Debian Bookworm (older) for builder to get GCC 12 instead of GCC 14
FROM python:3.11-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip==25.3

WORKDIR /build

# Step 1: Build and INSTALL TreeLite 4.6.1 from source with CMake config files
RUN git clone --depth 1 --branch 4.6.1 https://github.com/dmlc/treelite.git && \
    cd treelite && \
    mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/treelite \
      -DBUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install

# Step 2: Install TreeLite Python package pointing to our built library
WORKDIR /build/treelite/python
RUN CMAKE_PREFIX_PATH=/opt/treelite pip wheel --no-deps -w /wheels .

# Step 3: Now build tl2cgen 1.0.0 - it should find TreeLite via CMAKE_PREFIX_PATH
WORKDIR /build
RUN git clone --depth 1 --branch 1.0.0 https://github.com/dmlc/tl2cgen.git && \
    cd tl2cgen && \
    mkdir build && cd build && \
    CMAKE_PREFIX_PATH=/opt/treelite cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Step 4: Build tl2cgen Python wheel
WORKDIR /build/tl2cgen/python
RUN CMAKE_PREFIX_PATH=/opt/treelite pip wheel --no-deps -w /wheels .

# Main application stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (libgomp1 needed for TreeLite compiled .so files)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# FIX: GHSA-4xh5-x5gv-qwph
RUN pip install --upgrade pip==25.3

# Install all dependencies EXCEPT treelite and tl2cgen (we'll use our wheels)
RUN grep -v -E '^(treelite|tl2cgen)' requirements.txt > /tmp/requirements-filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements-filtered.txt

# Copy and install the TreeLite and tl2cgen wheels built with 4.6.1
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy TreeLite shared libraries to runtime image
COPY --from=builder /opt/treelite/lib/* /usr/local/lib/
RUN ldconfig

# Copy shared common code
COPY common/ ./common/

# Copy prediction server code
COPY prediction/ ./prediction/

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application using uvicorn
CMD ["python", "-m", "uvicorn", "prediction.prediction_server:app", "--host", "0.0.0.0", "--port", "8001"]
