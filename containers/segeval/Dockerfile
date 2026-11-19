# Segmentation-quality evaluation image — bolt3x/mirage-segeval:1.0.0
#
# Runs SEG_QUALITY_EVAL and MERGE_SEG_EVAL: reference-free 2D cell-segmentation quality
# with the vendored CellSegmentationEvaluator 1.5.19 subset in bin/utils/cse/. Scores the
# cell and nucleus masks against their image and emits a composite QualityScore.
#
# CPU only. The cost is RAM (per-pixel structures over the decompressed masks), hence the
# image-sized memory in conf/modules.config and --cse_max_pixels.
#
# Opt-in (skip_seg_quality_eval defaults to true).
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

LABEL description="Mirage SEG_QUALITY_EVAL: vendored CellSegmentationEvaluator (2D) metrics"

# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its
# script. python:*-slim ships without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Build context is the repository root. segeval.txt's `-c constraints.txt` resolves next
# to it, so both land in one directory.
COPY requirements/constraints.txt requirements/segeval.txt /tmp/requirements/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements/segeval.txt

# smoke.sh imports this image's stack so a broken resolve fails the build;
# containers.yml runs it again on the built image.
COPY containers/segeval/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# Cache directories under /tmp: the cluster's $HOME is read-only inside the container.
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
