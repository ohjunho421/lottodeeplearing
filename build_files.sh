#!/bin/bash

# Set memory-efficient pip options
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies with memory optimization
pip install --no-cache-dir -r requirements_basic.txt

# Remove unnecessary files to reduce Lambda size
find /opt/venv -name "*.pyc" -delete
find /opt/venv -name "__pycache__" -type d -exec rm -rf {} +
find /opt/venv -name "*.pyo" -delete
find /opt/venv -name "tests" -type d -exec rm -rf {} +
find /opt/venv -name "test" -type d -exec rm -rf {} +

# Create necessary directories
mkdir -p static
mkdir -p staticfiles_build

# Collect static files
python manage.py collectstatic --noinput --clear

# Copy static files to build directory
if [ -d "staticfiles" ]; then
    cp -r staticfiles/* staticfiles_build/
fi