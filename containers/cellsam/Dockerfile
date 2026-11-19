# torch 2.6.0 is the first release fixing CVE-2025-32434 (torch.load RCE), reachable here
# because cellSAM downloads model weights. 2.6.0 has no cuda12.1 build; 12.4 is the nearest.
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its script.
RUN apt-get update && apt-get install -y --no-install-recommends git procps \
    && rm -rf /var/lib/apt/lists/*

# Build context is the repository root. Inline pins below must agree with
# requirements/constraints.txt, or the build fails.
COPY requirements/constraints.txt /tmp/requirements/constraints.txt

RUN python -m pip install --upgrade pip

# dask and scikit-image are named although cellSAM may pull them in: segment_cellsam.py
# imports both, and the harmonisation test only sees packages named here.
RUN pip install -c /tmp/requirements/constraints.txt \
    numpy==1.26.4 \
    tifffile==2025.5.10 \
    zarr==2.18.3 \
    numcodecs==0.13.1 \
    dask==2026.7.1 \
    scikit-image==0.25.2

RUN python -m pip install -c /tmp/requirements/constraints.txt "imagecodecs==2025.3.30"

# Pinned: an unpinned git install makes every rebuild a different image.
RUN pip install -c /tmp/requirements/constraints.txt git+https://github.com/vanvalenlab/cellSAM.git@362a8e521bcd7313d5cb280bcd4cfc12977c1e9f

# Cache directories under /tmp: tasks run as the host user and the cluster's $HOME is
# read-only, so a library writing to ~/.cache or ~/.config would crash.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# smoke.sh imports what this image's bin/ scripts import, so a broken stack fails the
# build; containers.yml runs it again on the built image. Kept last so the COPY does not
# invalidate earlier layers.
COPY containers/cellsam/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
