# Use Miniconda3 as base image
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_COLOR=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        postgresql-server-dev-all \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libfreetype6-dev \
        pkg-config \
        libtk8.6 \
        tk-dev \
        python3-tk \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 5006

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
