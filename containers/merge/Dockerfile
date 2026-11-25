# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM cdgatenbee/valis-wsi:1.2.0@sha256:4fd20b687ae0f08c47011753bc5083dca5ac65a454d3615589eddf6b14e3c419

# Install required native dependencies for imagecodecs
# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its script.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        libopenjp2-7-dev \
        libzstd-dev \
        libwebp-dev \
        libtiff-dev \
        libpng-dev \
        liblzma-dev \
        zlib1g-dev \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Build context is the repository root. Inline pins below must agree with
# requirements/constraints.txt, or the build fails.
COPY requirements/constraints.txt /tmp/requirements/constraints.txt

# No numba: neither script imports it, and the VALIS base already ships it.
RUN pip install --no-cache-dir -c /tmp/requirements/constraints.txt \
        numpy==1.26.4 \
        tifffile==2025.5.10 \
        imagecodecs==2025.3.30 \
        opencv-python==4.10.0.82 \
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
COPY containers/merge/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
