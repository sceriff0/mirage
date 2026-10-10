# PALOM image for the registration benchmark -- bolt3x/mirage-palom:1.0.0
#
# NOT A PIPELINE IMAGE. No modules/local process runs here. It runs one competitor of the
# registration benchmark, `python -m benchmarks.regbench run --method palom`
# (benchmarks/regbench/methods/palom.py; rows `palom` and `palom_b1024`), with the
# repository bind-mounted. The benchmark's rule is that a method runs as its authors
# publish it, so this is PALOM's release from PyPI on the stack its own metadata asks for.
#
# CPU only, no JVM, no learned weights: PALOM is numpy / scikit-image / OpenCV / dask.
#
# PYTHON 3.10, NOT THE 3.11 OF containers/stare. PALOM requires scikit-image<0.20, and
# scikit-image 0.19.3 publishes wheels up to CPython 3.10 only; PALOM's own lock file
# (conda-env/env-win-64.lock at v2026.4.1) is Python 3.10 too.
#
# DIGEST-PINNED (ruling R6; tests/test_base_images_are_digest_pinned.py). The digest is
# the multi-platform index Docker Hub served for `python:3.10-slim` on 2026-10-10 (Python
# 3.10.22 on Debian trixie), read from the registry's API. The tag stays for readability.
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

# procps: Nextflow's task-metrics wrapper needs `ps`, and although Nextflow does not run
# this image, every image in this directory carries it so the rule has no exception to
# remember (containers/README.md, "Runtime requirements every image must satisfy").
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# The build context is the REPOSITORY ROOT. This image takes NO constraints file:
# requirements/palom.txt is a complete pin file whose header says why (PALOM caps
# scikit-image and tifffile below the harmonised versions), and the divergences are
# recorded in tests/test_container_harmonisation.py's PINNED_EXCEPTIONS.
COPY requirements/palom.txt requirements/stare.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/palom.txt

# STARE, WITHOUT ITS DEPENDENCIES. The benchmark reads every image through
# stare.slide_io and stare.ome, in every method's image. STARE's metadata requires
# scikit-image>=0.24, which PALOM forbids, so a plain install cannot resolve; the two
# modules the benchmark uses import numpy, tifffile and zarr only, all pinned in
# palom.txt. The requirement line is CUT OUT OF requirements/stare.txt rather than
# written here, so STARE's version is still written exactly once in this repository.
# `test -s` makes a stare.txt that no longer has that line fail the build here.
RUN grep -E '^stare-registration @ ' /tmp/requirements/stare.txt > /tmp/requirements/stare-only.txt \
    && test -s /tmp/requirements/stare-only.txt \
    && pip install --no-cache-dir --no-deps -r /tmp/requirements/stare-only.txt

# The final verification: the same smoke.sh .github/workflows/containers.yml runs with
# `docker run` against the built image, so one definition runs at build time and in CI.
COPY containers/palom/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

ENV PYTHONUNBUFFERED=1

# Read-only $HOME safety, the same three variables the other images set: matplotlib
# (a PALOM requirement) writes ~/.config/matplotlib on first import, and the cluster's
# $HOME is a read-only mount.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

CMD ["python", "--version"]

# OCI provenance, last so that VCS_REF invalidates one metadata layer only.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
