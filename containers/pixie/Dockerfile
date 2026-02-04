# Pixie Container for ATEIA Pipeline
# Based on ark-analysis 0.6.4 (the version Pixie is pinned to)
#
# Build: docker build -t bolt3x/attend_image_analysis:pixie .
# Push:  docker push bolt3x/attend_image_analysis:pixie

FROM python:3.9-slim

LABEL maintainer="ATEIA Pipeline"
LABEL description="Pixie clustering container with ark-analysis 0.6.4"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# ark-analysis 0.6.4 is the version Pixie is pinned to
RUN pip install --no-cache-dir \
    ark-analysis==0.6.4 \
    alpineer \
    feather-format \
    pandas>=1.5.0 \
    numpy>=1.23.0 \
    scikit-learn>=1.1.0 \
    scikit-image>=0.19.0 \
    tifffile>=2022.5.4 \
    matplotlib>=3.5.0 \
    xarray>=2022.3.0 \
    scipy>=1.9.0

# Create bin directory and copy scripts
RUN mkdir -p /usr/local/bin

# Copy clustering scripts
COPY bin/pixie_pixel_cluster.py /usr/local/bin/
COPY bin/pixie_cell_cluster.py /usr/local/bin/

# Make scripts executable
RUN chmod +x /usr/local/bin/pixie_pixel_cluster.py \
    && chmod +x /usr/local/bin/pixie_cell_cluster.py

# Set working directory
WORKDIR /data

# Default command
CMD ["python", "--version"]
