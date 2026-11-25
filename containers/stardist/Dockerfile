# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM tensorflow/tensorflow:2.21.0-gpu-jupyter@sha256:742a91e780b2d1d51a3352d1daf71415a73cc50594a3d7c0110d5f9949b7435b

WORKDIR /workspace

ARG NVIDIA_DRIVER_VERSION=470

# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its script.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    libnvidia-compute-${NVIDIA_DRIVER_VERSION} \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build context is the repository root. Inline pins below must agree with
# requirements/constraints.txt, or the build fails.
COPY requirements/constraints.txt /tmp/requirements/constraints.txt

# csbdeep and dask are named although stardist pulls csbdeep in: segment.py imports both.
# csbdeep is pinned here, not in constraints.txt, because only this image installs it.
RUN pip install --no-cache-dir -c /tmp/requirements/constraints.txt \
    stardist==0.9.1 \
    csbdeep==0.8.2 \
    dask==2026.7.1 \
    numpy==1.26.4 \
    scikit-image==0.25.2 \
    scipy==1.15.3 \
    tifffile==2025.5.10 \
    imagecodecs==2025.3.30 \
    zarr==2.18.3 \
    numcodecs==0.13.1

# StarDist's optional accelerators: stardist uses `edt` (distance transform) and
# `gputools` (OpenCL non-maximum suppression, hence the OpenCL apt packages) when
# importable. Pinned so rebuilds are reproducible.
RUN pip install -c /tmp/requirements/constraints.txt gputools==0.3.1 edt==3.1.2

# Re-assert the exact pin: gputools/edt above can pull a newer numpy transitively.
RUN pip install --no-cache-dir -c /tmp/requirements/constraints.txt "numpy==1.26.4"

# Cache directories under /tmp: tasks run as the host user and the cluster's $HOME is
# read-only, so a library writing to ~/.cache or ~/.config would crash.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# smoke.sh imports what this image's bin/ scripts import, so a broken stack fails the
# build; containers.yml runs it again on the built image. Kept last so the COPY does not
# invalidate earlier layers.
COPY containers/stardist/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
