# Use slim Python 3.11 Bookworm image
FROM python:3.11-slim-bookworm

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including graph-tool from default repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    python3-graph-tool \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    libtk8.6 \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install wheel and setuptools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with verbose output
RUN pip install --no-cache-dir -v -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 5006 available
EXPOSE 5006

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "run.py"]
