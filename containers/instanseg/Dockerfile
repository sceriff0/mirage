# torch 2.6.0 carries the CVE-2025-32434 fix.
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime@sha256:2428b92ebbaeceba5572b98c18c8a94e43162bead6e88588ad54471147c58a20

LABEL maintainer="MIRAGE Pipeline"
# Keep the versions in this label in step with the FROM line.
LABEL description="InstanSeg segmentation container (PyTorch 2.6.0 + CUDA 11.8)"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# Build context is the repository root. Inline pins below must agree with
# requirements/constraints.txt, or the build fails.
COPY requirements/constraints.txt /tmp/requirements/constraints.txt

# OpenCL ICD loader and headers, for downstream tools; InstanSeg itself uses PyTorch/CUDA.
# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its script.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -c /tmp/requirements/constraints.txt \
    instanseg-torch==0.1.1 \
    imagecodecs==2025.3.30 \
    numpy==1.26.4 \
    tifffile==2025.5.10 \
    zarr==2.18.3 \
    numcodecs==0.13.1

# Cache directories under /tmp: tasks run as the host user and the cluster's $HOME is
# read-only, so a library writing to ~/.cache or ~/.config would crash.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# smoke.sh imports what this image's bin/ scripts import, so a broken stack fails the
# build; containers.yml runs it again on the built image. Kept last so the COPY does not
# invalidate earlier layers.
COPY containers/instanseg/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
