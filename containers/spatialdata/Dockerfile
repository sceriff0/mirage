# SpatialData export image — bolt3x/mirage-spatialdata:1.0.0
#
# Runs EXPORT_SPATIALDATA: serializes postprocessing output into a scverse-native
# SpatialData .zarr (labels + shapes + AnnData table, optionally the pyramid), and
# the JOIN_FLOWPATH attach step that folds QuPath/FlowPath gating results back in.
#
# CPU only. geopandas/shapely need libgeos at runtime.
#
# spatialdata >= 0.8.0 needs zarr >= 3 and Python >= 3.12, while the other images pin
# zarr 2.18 for tifffile's aszarr reads; hence a separate image. On an older Python pip
# reports "no matching distribution found", not a version error.
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

LABEL description="Mirage EXPORT_SPATIALDATA: SpatialData/AnnData/OME-NGFF serialization"

# libgeos-c1v5 -> shapely/geopandas; libexpat/procps are standard Nextflow runtime needs
# (procps supplies `ps`, which Nextflow calls to collect task metrics).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-c1v5 \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Build context is the repository root. No constraints.txt: this image's zarr 3 stack
# cannot take the harmonised pins.
COPY requirements/spatialdata.txt /tmp/requirements/spatialdata.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements/spatialdata.txt

# smoke.sh imports the stack (spatialdata -> zarr 3 -> ome_zarr -> dask, plus GEOS) so a
# broken resolve fails the build; containers.yml runs it again on the built image.
COPY containers/spatialdata/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# Cache directories under /tmp: the cluster's $HOME is read-only.
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
