# SpatialData export image — bolt3x/mirage-spatialdata:1.0.0
#
# Runs EXPORT_SPATIALDATA: serializes postprocessing output into a scverse-native
# SpatialData .zarr (labels + shapes + AnnData table, optionally the pyramid), and
# the JOIN_FLOWPATH attach step that folds QuPath/FlowPath gating results back in.
#
# CPU-only scientific-Python stack — no JVM, no GPU, no libvips. The heaviest thing
# here is the geopandas/shapely GEOS stack, which needs libgeos at runtime.
#
# NOTE the zarr major version: spatialdata >= 0.8.0 requires zarr >= 3, whereas the
# `tiled` image pins zarr 2.18 for tifffile's aszarr region reads. They are separate
# images precisely so those two constraints never have to be reconciled.
#
# Python 3.12, NOT the 3.11 the other CPU images (`tiled`) use:
# spatialdata 0.8.0 declares Requires-Python >= 3.12. On 3.11 pip silently skips it
# and reports "no matching distribution found for spatialdata>=0.8.0", which reads
# like a typo'd package name rather than a Python-version floor.
# DIGEST-PINNED (ruling R6). The tag stays for readability; the digest is what makes the
# build reproducible -- a version tag is republished, so rebuilding from the tag alone can
# produce a different image than the one that generated the published results. Resolved
# 2026-09-02; Dependabot updates the digest in place. Guarded by
# tests/test_base_images_are_digest_pinned.py.
FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

LABEL description="Mirage EXPORT_SPATIALDATA: SpatialData/AnnData/OME-NGFF serialization"

# libgeos-c1v5 -> shapely/geopandas; libexpat/procps are standard Nextflow runtime needs
# (procps supplies `ps`, which Nextflow calls to collect task metrics).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-c1v5 \
        procps \
    && rm -rf /var/lib/apt/lists/*

# The build context is the REPOSITORY ROOT (containers.yml passes `context: .`).
# NO constraints.txt here, deliberately: this is the documented python:3.12 / zarr>=3
# island, and requirements/spatialdata.txt carries no `-c` line for the same reason.
COPY requirements/spatialdata.txt /tmp/requirements/spatialdata.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements/spatialdata.txt

# Fail the BUILD rather than a cluster run: the import graph here (spatialdata ->
# zarr 3 -> ome_zarr -> dask, plus the GEOS shared library) is exactly the kind that
# resolves in pip and then explodes on first import.
#
# THE CHECK ITSELF MOVED, VERBATIM, into containers/spatialdata/smoke.sh (CI redesign
# Phase 6, 2026-09-01), so the same definition runs here at build time AND from
# .github/workflows/containers.yml with `docker run` against the built image.
COPY containers/spatialdata/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# Read-only $HOME safety: several of these libraries default their caches to ~/.cache.
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
