# STARE registration image — bolt3x/mirage-stare:1.0.0
#
# STARE (Scalable Tile-parallel Alignment by Robust Estimation) is the fully
# parallel, JVM-free registration method behind registration_method='tiled'. This image runs
# TILED_COARSE, TILED_REG_TILE, TILED_SOLVE, TILED_STITCH and WARP_SEG_QC's tiled backend
# (lib/WarpBackends.groovy). The method is the `stare` package (github.com/sceriff0/stare), installed
# below; bin/tiled_*.py are shims over its stages.
#
# SLIM, AND THAT IS THE POINT OF A NEW IMAGE. It replaces bolt3x/mirage-tiled:1.0.0, which
# carried two CPU wheels (torch, kornia; ~1 GB), libgl1/libglib2.0-0 for them, and the DISK +
# LightGlue checkpoints baked under TORCH_HOME=/opt/torch -- all for a COARSE front-end that
# was retired on 2026-09-27 (an FFT NCC rotation sweep with a scikit-image ORB fallback now;
# stare/coarse_align.py imports neither, tests/test_container_harmonisation.py's
# test_tiled_coarse_imports_no_learned_stack pins that). Every stage is numpy / scipy /
# scikit-image / tifffile / zarr. smoke.sh asserts torch is NOT importable here, so a wheel
# that sneaks back in (a transitive pull) fails the build rather than growing the image.
# No Java/BioFormats, no libvips, no GPU, no learned weights.
#
# DIGEST-PINNED (ruling R6), the SAME digest containers/tiled used. The tag stays for
# readability; the digest is what makes the build reproducible -- a version tag is
# republished, so rebuilding from the tag alone can produce a different image than the one
# that generated the published results. Dependabot updates the digest in place. Guarded by
# tests/test_base_images_are_digest_pinned.py.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# scikit-image / scipy import cleanly on slim with no extra *Python-facing* system libs, but
# `procps` is not optional: Nextflow's task-metrics wrapper hard-exits BEFORE the script block
# when `ps` is absent -- nextflow/executor/command-trace.txt does
#     command -v ps &>/dev/null || { >&2 echo "Command 'ps' required by nextflow ..."; exit 1; }
# so every task in this image would fail with exit status 1 and empty stdout as soon as the
# run collected metrics (-with-report / -with-trace, i.e. any run producing the resource
# report). python:*-slim ships no procps. Same reason it is installed in
# containers/spatialdata/Dockerfile. (libgl1 / libglib2.0-0 are gone with torch/kornia: they
# were there only because those wheels dlopen libGL at import time.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# The build context is the REPOSITORY ROOT (containers.yml passes `context: .`), so the
# pin files come from requirements/ -- the same files .github/workflows/*.yml install. There
# is no second copy of any version string to keep in step (stare.txt was tiled.txt until the
# image was renamed). stare.txt opens with `-c constraints.txt`, which pip resolves next to it.
COPY requirements/constraints.txt requirements/stare.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/stare.txt

# THE METHOD ITSELF comes with the requirements above: stare.txt pins `stare-registration`
# to a release of github.com/sceriff0/stare, the method's only source since v1.0.0, so the
# image no longer copies anything of STARE out of this repository. The bin/tiled_*.py
# scripts the modules invoke are shims over it, so every TILED_* task fails at import
# without it -- smoke.sh imports it. Guarded by tests/test_container_harmonisation.py and
# tests/test_stare_image_is_slim.py.

# The final verification: the same smoke.sh .github/workflows/containers.yml runs with
# `docker run` against the built image, so one definition runs at build time and in CI.
COPY containers/stare/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# Unbuffered stdout: a long COARSE or STITCH task's progress lines reach Nextflow's
# .command.out as they happen rather than at exit.
ENV PYTHONUNBUFFERED=1

# Read-only $HOME safety, the same three variables containers/segeval and
# containers/spatialdata set. Several of these libraries default their caches to ~/.cache;
# nextflow.config's docker profile runs tasks with `-u $(id -u):$(id -g)` and the cluster's
# $HOME is a read-only mount, so a cache write turns into a crash far from its cause -- the
# same class of failure as the VALIS JVM/jgo cache. MPLCONFIGDIR is not decoration:
# matplotlib writes ~/.config/matplotlib on its first import if anything pulls it in.
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
