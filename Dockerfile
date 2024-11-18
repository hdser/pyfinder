# Use Miniconda3 as base image
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
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

# Install the Skewed Keyring package for graph-tool
RUN wget https://downloads.skewed.de/skewed-keyring/skewed-keyring_1.1_all_$(lsb_release -cs).deb -O /tmp/skewed-keyring.deb && \
    dpkg -i /tmp/skewed-keyring.deb || apt-get install -f -y && \
    rm /tmp/skewed-keyring.deb

# Add the graph-tool repository to the sources list
RUN echo "deb [signed-by=/usr/share/keyrings/skewed-keyring.gpg] https://downloads.skewed.de/apt $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/graph-tool.list

# Update APT sources and install python3-graph-tool along with other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-graph-tool \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy Conda environment file
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose port and run application
EXPOSE 5006

CMD ["conda", "run", "-n", "myenv", "python", "run.py"]



# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Define build arguments for pip (optional, can be customized)
ARG PIP_TIMEOUT=100
ARG PIP_RETRIES=10

# Update and install initial system dependencies, including tkinter
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
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

# Install the Skewed Keyring package for graph-tool
RUN wget https://downloads.skewed.de/skewed-keyring/skewed-keyring_1.1_all_$(lsb_release -cs).deb -O /tmp/skewed-keyring.deb && \
    dpkg -i /tmp/skewed-keyring.deb || apt-get install -f -y && \
    rm /tmp/skewed-keyring.deb

# Add the graph-tool repository to the sources list
RUN echo "deb [signed-by=/usr/share/keyrings/skewed-keyring.gpg] https://downloads.skewed.de/apt $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/graph-tool.list

# Update APT sources and install python3-graph-tool along with other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-graph-tool \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Remove 'graph-tools' or 'graph-tool' from requirements.txt if present
RUN sed -i '/graph-tools/d' requirements.txt
RUN sed -i '/graph-tool/d' requirements.txt

# Create pip configuration file to set timeout and retries
RUN mkdir -p /etc && \
    echo "[global]" > /etc/pip.conf && \
    echo "timeout = ${PIP_TIMEOUT}" >> /etc/pip.conf && \
    echo "retries = ${PIP_RETRIES}" >> /etc/pip.conf

# Upgrade pip and install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user and set ownership
RUN useradd -m pyfinderuser && \
    chown -R pyfinderuser:pyfinderuser /app

# Switch to the non-root user
USER pyfinderuser

# Expose the application port
EXPOSE 5006

# Command to run the application
CMD ["python3", "run.py"]
