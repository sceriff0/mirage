# VALIS 1.2.0 for the pipeline -- bolt3x/mirage-valis:1.0.0 (REGISTER, WARP_SEG_QC's VALIS path)
#
# UPSTREAM'S IMAGE, MADE RUNNABLE BY A USER WHO IS NOT ROOT, AND NOTHING ELSE. MIRAGE's
# VALIS mode is valis-wsi 1.2.0 at its defaults (bin/register.py), and that is upstream's
# own build: nothing below reinstalls or adds to VALIS, torch, Kornia, pyvips, libvips, the
# JVM, Bio-Formats or any Python package. smoke.sh fails the build if the pip freeze differs
# from the base image's by a single line.
#
# WHY IT CANNOT BE THE BARE UPSTREAM DIGEST. Upstream's image is built to run as root under
# Docker. Apptainer, which the cluster uses, runs it as the calling user with the image
# read-only and another $HOME, and four things break (place_assets.py has the reading):
#   1. NO PYTHON: the interpreter and every package are under /root (mode 0700).
#      `chmod a+rx /root` below.
#   2. NO BIO-FORMATS JAR beside the INSTALLED package, so VALIS would ask Maven. Linked.
#   3. NO WEIGHTS for another user: they sit in root's torch cache. TORCH_HOME=/opt/torch,
#      linked to upstream's directory.
#   4. <package>/data/bf_formats.txt is REWRITTEN on every JVM start (new in 1.2.0). Made
#      writable; under Apptainer the image is still read-only, so REGISTER and WARP_SEG_QC
#      run with `--writable-tmpfs` (conf/modules.config, singularity/apptainer only).
# These are the same four fixes as the registration benchmark's bolt3x/mirage-valis120,
# without that image's additions (zarr, imagecodecs, STARE's reader, pytest, a tifffile
# downgrade), which the pipeline does not need.
#
# DIGEST-PINNED (ruling R6; tests/test_base_images_are_digest_pinned.py): the multi-platform
# index Docker Hub served for the tag on 2026-10-11, the same pin as containers/valis120.
FROM cdgatenbee/valis-wsi:1.2.0@sha256:4fd20b687ae0f08c47011753bc5083dca5ac65a454d3615589eddf6b14e3c419

# procps: Nextflow's task-metrics wrapper needs `ps` (Ubuntu 24.04 already has it; written
# out because every image here names it -- containers/README.md).
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Point 1. Nothing private is in /root: it is a published image.
RUN chmod a+rx /root

# The freeze of upstream's stack, kept in the image: smoke.sh compares the final stack with it.
RUN mkdir -p /opt/mirage \
    && python -m pip freeze --all | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' | LC_ALL=C sort > /opt/mirage/upstream-freeze.txt \
    && test "$(wc -l < /opt/mirage/upstream-freeze.txt)" -ge 85 \
    && chmod -R a+rX /opt/mirage

# Point 3 (set before place_assets.py, which refuses to run without it) and the read-only-
# $HOME variables every image here sets; matplotlib is imported by VALIS.
ENV TORCH_HOME=/opt/torch \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# Points 2 to 4: the jar, the weights and the permissions of <package>/data. No network.
COPY containers/valis/place_assets.py /usr/local/bin/mirage-valis-place-assets.py
RUN python /usr/local/bin/mirage-valis-place-assets.py

# The proof of points 1 to 4, as an unprivileged user with the network closed. The same
# script runs again through `docker run` in .github/workflows/containers.yml.
COPY containers/valis/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

ENV PYTHONUNBUFFERED=1

CMD ["python", "--version"]

# OCI provenance, last so that VCS_REF invalidates one metadata layer only.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
