FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    unzip \
    openjdk-11-jre-headless \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Install nextflow
RUN wget -qO- https://get.nextflow.io | bash \
    && mv nextflow /usr/local/bin/nextflow

COPY . /app

ENTRYPOINT ["/usr/local/bin/nextflow"]
