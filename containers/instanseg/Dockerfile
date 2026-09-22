# torch 2.6.0 carries the CVE-2025-32434 fix; cuda11.8 is unchanged.
# DIGEST-PINNED (ruling R6). The tag stays for readability; the digest is what makes the
# build reproducible -- a version tag is republished, so rebuilding from the tag alone can
# produce a different image than the one that generated the published results. Resolved
# 2026-09-02; Dependabot updates the digest in place. Guarded by
# tests/test_base_images_are_digest_pinned.py.
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime@sha256:8d409f72f99e5968b5c4c9396a21f4b723982cfdf2c1a5b9cc045c5d0a7345a1

LABEL maintainer="MIRAGE Pipeline"
# The version in this label is the FROM's, and it has been wrong before: it said
# "PyTorch 2.5.1 + CUDA 12.4" against a 2.6.0-cuda11.8 base -- three numbers, two wrong.
LABEL description="InstanSeg segmentation container (PyTorch 2.6.0 + CUDA 11.8)"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# The build context is the REPOSITORY ROOT (containers.yml passes `context: .`), so the
# harmonised pins come from requirements/constraints.txt -- the SAME file the workflows and
# the other images read. The inline pins below must AGREE with it: a constraint does not
# override a divergent direct requirement, it conflicts with it and fails the build, which
# is exactly the behaviour wanted.
COPY requirements/constraints.txt /tmp/requirements/constraints.txt

# OpenCL ICD loader and headers (kept in case downstream tools need OpenCL;
# unrelated to InstanSeg itself, which uses PyTorch/CUDA).
# procps is not optional. Nextflow's task-metrics wrapper hard-exits BEFORE the script
# block when `ps` is absent, and params.enable_trace defaults to true, so every task in
# this image would fail with exit status 1 and empty stdout. Installed in all eleven
# images rather than "verified present in this base" -- apt is a no-op where it already
# is, and containers/<dir>/smoke.sh asserts it either way.
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

# Fail the BUILD rather than a cluster run. containers/<dir>/smoke.sh imports exactly
# the third-party modules this image's bin/ scripts import at run time, so a stack that
# resolved in pip and then explodes on first import stops here. The SAME script is run
# again by .github/workflows/containers.yml against the built image, so there is one
# definition exercised in two places rather than two copies that drift.
#
# LAST, and after every pip install, on purpose: a COPY invalidates the layer cache for
# everything below it, so putting this at the end costs no rebuilds.
# Read-only $HOME safety, the same three variables containers/segeval and
# containers/spatialdata set. Several of these libraries default their caches to ~/.cache;
# nextflow.config's docker profile runs tasks with `-u $(id -u):$(id -g)` and the cluster's
# $HOME is a read-only mount, so a cache write turns into a crash far from its cause -- the
# same class of failure as the VALIS JVM/jgo cache. MPLCONFIGDIR is not decoration here:
# matplotlib writes ~/.config/matplotlib on its first import.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

COPY containers/instanseg/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance. LAST IN THE FILE ON PURPOSE: VCS_REF changes on every commit, and an ARG
# referenced earlier would invalidate the layer cache for everything below it on every
# build. Here it invalidates one metadata layer. .github/workflows/containers.yml passes
# both values; a local `docker build` gets "unknown", which is honest.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
