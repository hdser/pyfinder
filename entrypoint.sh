#!/bin/bash

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate pyfinder

# Run the application
exec python -u run.py
