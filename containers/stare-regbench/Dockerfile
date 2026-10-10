# The registration benchmark's own STARE image -- bolt3x/mirage-stare-regbench:1.0.0
#
# NOT A PIPELINE IMAGE. No modules/local process runs here; the pipeline's STARE image is
# containers/stare (bolt3x/mirage-stare), which is left exactly as it is. This one runs
# three things of benchmarks/regbench, with the repository bind-mounted:
#   - the `stare` run:  python -m benchmarks.regbench run --method stare
#   - the tissue step:  python -m benchmarks.regbench tissue
#   - the scorer:       python -m benchmarks.regbench score
#
# WHY A SECOND IMAGE FOR THE SAME METHOD (ruling R3.9a, 2026-10-10). The benchmark reports
# every method's peak memory, and reads it through psutil. containers/stare has no psutil
# (the pipeline does not need it), so there the benchmark's monitor falls back to /proc
# and STARE would be the one method measured by another instrument. This image adds
# psutil, pytest (the first cluster command runs the benchmark's own tests inside each
# image) and OpenCV for the tissue-level block. THE METHOD IS THE SAME: the same base
# image at the same digest, and requirements/stare-regbench.txt `-r`-includes
# requirements/stare.txt, so STARE's pinned release and its numeric stack are the
# pipeline image's, line for line.
#
# ONE OPENCV FOR A WHOLE RESULT. The tissue-level numbers are OpenCV's ORB and RANSAC, so
# they are computed in one place: `tissue` and `score` run in this image, on the
# opencv-python-headless of requirements/constraints.txt (4.10.0.82), whatever OpenCV a
# method's own image happens to carry.
#
# Still CPU-only, no JVM, no libvips, no torch: smoke.sh asserts torch does not import.
#
# DIGEST-PINNED (ruling R6; tests/test_base_images_are_digest_pinned.py): the SAME base
# and digest as containers/stare.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# procps: Nextflow's task-metrics wrapper needs `ps`, and although Nextflow does not run
# this image, every image in this directory carries it so the rule has no exception to
# remember (containers/README.md, "Runtime requirements every image must satisfy").
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# The build context is the REPOSITORY ROOT. stare-regbench.txt opens with
# `-c constraints.txt` and `-r stare.txt`, which pip resolves next to it, so the three are
# copied into one directory.
COPY requirements/constraints.txt requirements/stare.txt requirements/stare-regbench.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/stare-regbench.txt

# The final verification: the same smoke.sh .github/workflows/containers.yml runs with
# `docker run` against the built image, so one definition runs at build time and in CI.
COPY containers/stare-regbench/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

ENV PYTHONUNBUFFERED=1

# Read-only $HOME safety, the same three variables the other images set.
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
