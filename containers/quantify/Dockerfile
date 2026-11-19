
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04@sha256:ae8a022c02aec945c4f8c52f65deaf535de7abb58e840350d19391ec683f4980

# Install Python and essential tools
# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its script.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Build context is the repository root. Inline pins below must agree with
# requirements/constraints.txt, or the build fails.
COPY requirements/constraints.txt /tmp/requirements/constraints.txt

# Only what this image's scripts import: matplotlib, numpy, pandas, scipy, skimage, tifffile.
RUN pip install --no-cache-dir -c /tmp/requirements/constraints.txt \
    numpy==1.26.4 \
    pandas==2.3.3 \
    scikit-image==0.25.2 \
    scipy==1.15.3 \
    matplotlib==3.9.2 \
    tifffile==2025.5.10 \
    imagecodecs==2025.3.30 \
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
COPY containers/quantify/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
