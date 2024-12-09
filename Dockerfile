# Use Miniconda3 as base image
FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_COLOR=1 \
    PORT=5006 \
    HOST=0.0.0.0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        postgresql-server-dev-all \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy environment file first to leverage Docker cache
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Copy data files
COPY data /app/data

# Create startup script
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate pyfinder\n\
exec python -u run_mini.py\n'\
> /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 5006

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5006/ || exit 1

# Run the application
CMD ["/app/start.sh"]