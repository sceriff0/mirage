# STARE registration image — bolt3x/mirage-stare:1.0.0
#
# STARE (Scalable Tile-parallel Alignment by Robust Estimation), the JVM-free method behind
# registration_method='tiled'. Runs TILED_COARSE, TILED_REG_TILE, TILED_SOLVE, TILED_STITCH
# and WARP_SEG_QC's tiled backend. The method is the `stare` package (github.com/sceriff0/stare);
# bin/tiled_*.py are shims over its stages.
#
# Slim: every stage needs only numpy / scipy / scikit-image / tifffile / zarr. No Java, no
# libvips, no GPU, no learned weights; smoke.sh fails the build if torch becomes importable.
#
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its
# script. python:*-slim ships without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Build context is the repository root; CI installs the same requirements files.
# stare.txt's `-c constraints.txt` resolves next to it.
COPY requirements/constraints.txt requirements/stare.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/stare.txt

# stare.txt also installs `stare-registration` at a pinned release; the bin/tiled_*.py
# shims fail at import without it.

# smoke.sh runs here and again in containers.yml against the built image.
COPY containers/stare/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# Unbuffered stdout so progress lines reach .command.out as they happen.
ENV PYTHONUNBUFFERED=1

# Cache directories under /tmp: tasks run as the host user and the cluster's $HOME is
# read-only, so a library writing to ~/.cache or ~/.config would crash.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

CMD ["python", "--version"]

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
