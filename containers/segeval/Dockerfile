# Segmentation-quality evaluation image — bolt3x/mirage-segeval:1.0.0
#
# Runs SEG_QUALITY_EVAL and MERGE_SEG_EVAL: reference-free 2D cell-segmentation
# quality scoring with the vendored CellSegmentationEvaluator subset under
# bin/utils/cse/ (upstream 1.5.19, Chen & Murphy, MBoC 2023). Scores a cell mask +
# nucleus mask against the multichannel image they were derived from and emits a
# composite QualityScore with NO ground-truth annotation.
#
# Pure CPU scientific-Python — no JVM, no BioFormats, no GPU, no libvips. The
# scorer's cost is RAM, not compute: it builds per-pixel index structures over the
# DECOMPRESSED masks, which is why conf/modules.config sizes the task off the image
# and why --cse_max_pixels exists to bin the input.
#
# OPT-IN in the pipeline (skip_seg_quality_eval defaults to true), so this image is
# only pulled by a run that asked for it.
# DIGEST-PINNED (ruling R6). The tag stays for readability; the digest is what makes the
# build reproducible -- a version tag is republished, so rebuilding from the tag alone can
# produce a different image than the one that generated the published results. Resolved
# 2026-09-02; Dependabot updates the digest in place. Guarded by
# tests/test_base_images_are_digest_pinned.py.
FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

LABEL description="Mirage SEG_QUALITY_EVAL: vendored CellSegmentationEvaluator (2D) metrics"

# procps is not optional. Nextflow's task-metrics wrapper hard-exits BEFORE the
# script block when `ps` is absent -- nextflow/executor/command-trace.txt does
#     command -v ps &>/dev/null || { >&2 echo "Command 'ps' required by nextflow ..."; exit 1; }
# so every task in this image would fail with exit status 1 and empty stdout as soon
# as the run collected metrics (-with-trace / -with-report -- i.e. every benchmark
# run, since benchmarks/configs/benchmark.config turns tracing on). python:*-slim
# ships no procps. Same reason it is installed in containers/tiled and spatialdata.
#
# NOTE: libgl1 / libglib2.0-0 used to be installed here. They are OpenCV's runtime
# deps, and nothing in this image imports cv2 -- the vendored CSE subset uses
# numpy/scipy/skimage/sklearn only. Removed rather than kept "just in case": an
# unexplained system package is indistinguishable from a load-bearing one later.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

# The build context is the REPOSITORY ROOT (containers.yml passes `context: .`).
# segeval.txt opens with `-c constraints.txt`, which pip resolves next to it, so both files
# must land in the same directory.
COPY requirements/constraints.txt requirements/segeval.txt /tmp/requirements/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements/segeval.txt

# Fail the BUILD rather than a cluster run. This stack is exactly the kind that
# resolves in pip and then explodes on first import: scikit-image 0.25 against numpy 1.26
# is a pairing pip will happily accept and then break on. The previous version of this file
# had no smoke test and a stray `pip install "numpy<2.0"` AFTER the pinned requirements,
# which could silently re-resolve numpy behind the pin.
#
# THE CHECK ITSELF MOVED, VERBATIM, into containers/segeval/smoke.sh (CI redesign
# Phase 6, 2026-09-01). It used to be written here and nowhere else, so it ran at
# build time only -- and a build only happens when someone changes this directory.
# The script is now run twice from ONE definition: here, and by
# .github/workflows/containers.yml with `docker run` against the built image.
COPY containers/segeval/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# Read-only $HOME safety. Several of these libraries default their caches to
# ~/.cache; on the cluster $HOME is read-only inside the container and a cache
# write turns into a crash far from its cause (the same class of failure as the
# VALIS JVM/jgo cache -- see docs/usage.md).
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

CMD ["python", "--version"]

# OCI provenance. LAST IN THE FILE ON PURPOSE: VCS_REF changes on every commit, and an ARG
# referenced earlier would invalidate the layer cache for everything below it on every
# build. Here it invalidates one metadata layer. .github/workflows/containers.yml passes
# both values; a local `docker build` gets "unknown", which is honest.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
