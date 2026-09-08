# DIGEST-PINNED (ruling R6). The tag stays for readability; the digest is what makes the
# build reproducible -- a version tag is republished, so rebuilding from the tag alone can
# produce a different image than the one that generated the published results. Resolved
# 2026-09-02; Dependabot updates the digest in place. Guarded by
# tests/test_base_images_are_digest_pinned.py.
FROM tensorflow/tensorflow:2.21.0-gpu-jupyter@sha256:742a91e780b2d1d51a3352d1daf71415a73cc50594a3d7c0110d5f9949b7435b

WORKDIR /workspace

# Install system dependencies first (if needed)
ARG NVIDIA_DRIVER_VERSION=470

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
    libnvidia-compute-${NVIDIA_DRIVER_VERSION} \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with exact versions

# The build context is the REPOSITORY ROOT (containers.yml passes `context: .`), so the
# harmonised pins come from requirements/constraints.txt -- the SAME file the workflows and
# the other images read. The inline pins below must AGREE with it: a constraint does not
# override a divergent direct requirement, it conflicts with it and fails the build, which
# is exactly the behaviour wanted.
COPY requirements/constraints.txt /tmp/requirements/constraints.txt
#
# csbdeep and dask are named explicitly even though stardist already pulls csbdeep in
# transitively: segment.py imports both directly (`from csbdeep.utils import normalize`,
# `import dask.array as da` for the map_overlap-based tiled read), and an install nothing
# in this Dockerfile names is invisible to tests/test_container_harmonisation.py's
# "installs what it imports" guard -- and to anyone reading this file to see what the
# script actually needs.
#
# csbdeep is ALSO the reason this line is now the pin's only home: containers/regqc
# installed csbdeep too until 2026-09-02, so it lived in requirements/constraints.txt.
# With one installer left, constraints.txt is the wrong place (it is for SHARED packages)
# and the pin lives here.
#
# pandas was removed 2026-09-02: bin/segment.py does not import it, and every install here
# passes `-c constraints.txt`, so a transitive pandas is still held at 2.3.3.
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

# StarDist's OPTIONAL ACCELERATORS, and the reason they are pinned rather than dropped:
# stardist imports `edt` for the Euclidean distance transform and `gputools` for
# OpenCL-accelerated non-maximum suppression when either is importable, so they are
# reached through the stardist distribution and never by bin/segment.py. The apt block
# above installs the OpenCL ICD loader and headers for exactly this. Unpinned they made
# every rebuild a different image; 0.3.1 and 3.1.2 are the current releases (PyPI,
# 2026-09-02), so this pin changes nothing about today's resolve and everything about
# tomorrow's. Exempted in tests/test_no_uninstalled_or_unimported_packages.py with that
# reason.
RUN pip install -c /tmp/requirements/constraints.txt gputools==0.3.1 edt==3.1.2

# Re-assert the exact pin: gputools/edt above can pull a newer numpy transitively.
RUN pip install --no-cache-dir -c /tmp/requirements/constraints.txt "numpy==1.26.4"

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

COPY containers/stardist/smoke.sh /usr/local/bin/mirage-smoke.sh
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
