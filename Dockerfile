# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV PYTHONPATH=/app

# Install system dependencies and add Ubuntu 23.04's repository for newer libstdc++
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    gcc \
    g++ \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    libtk8.6 \
    tk-dev \
    curl \
    gnupg \
    && echo "deb http://ports.ubuntu.com/ubuntu-ports lunar main restricted universe multiverse" > /etc/apt/sources.list.d/lunar.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends -t lunar libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && rm /etc/apt/sources.list.d/lunar.list

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /root/miniconda3 && \
    rm ~/miniconda.sh

# Create conda environment
RUN conda create -n pyfinder python=3.11 -y

# Install graph-tool and other conda dependencies, including libstdcxx-ng
RUN conda run -n pyfinder conda install -c conda-forge \
    graph-tool \
    numpy \
    pandas \
    networkx \
    libstdcxx-ng \
    -y

# Set LD_LIBRARY_PATH to prioritize Conda's lib directory
ENV LD_LIBRARY_PATH="/root/miniconda3/envs/pyfinder/lib:${LD_LIBRARY_PATH}"

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN conda run -n pyfinder pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN sed -i '/graph-tool/d' requirements.txt
RUN conda run -n pyfinder pip install --no-cache-dir -v -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for the application
EXPOSE 5006

# Command to run the application
CMD ["conda", "run", "-n", "pyfinder", "python", "run.py"]
