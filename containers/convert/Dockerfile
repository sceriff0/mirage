# Bio-Formats reader image for CONVERT_IMAGE — bolt3x/mirage-convert:1.0.0
#
# Build from the repository root:
#   docker build -t bolt3x/mirage-convert:1.0.0 -f containers/convert/Dockerfile .
#
# Contents: eclipse-temurin:21-jre-jammy (JAVA_HOME=/opt/java/openjdk), python3.10, bioio
# 3.5.0 with six reader plugins including bioio-bioformats, and the Bio-Formats jars and
# JVM cache baked under /root.
#
# bioio-bioformats reads .svs, .qptiff, .vsi, .scn, .mrxs, .bif and .ims; the other plugins
# cover OME-TIFF, TIFF, ND2, CZI and LIF. bin/utils/ome_io.py's detect_reader() routes them.
#
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM eclipse-temurin:21-jre-jammy@sha256:eebd356ad7358b7094758e5787a6726f332917cfd56feab6457c56dab895cdbf

LABEL maintainer="mirage WSI pipeline"
LABEL description="Bio-Formats CONVERT_IMAGE: bioio + bioio-bioformats with baked Bio-Formats jars"

# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its script.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
        coreutils \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first: jammy's 22.0.2 resolver reports this package set's conflict opaquely.
RUN pip3 install --no-cache-dir --upgrade pip

# dask is pinned although bioio pulls it in unbounded: the read and the chunked write in
# bin/convert_image.py both go through dask.
#
# numpy and tifffile diverge from the harmonised set, both forced upstream:
#   * bioio-ome-tiff 1.4.0 requires tifffile[zarr]<2025.1.10 on python_version < "3.11"
#     (this base is 3.10); 2024.12.12 is the newest that satisfies it.
#   * the bioio 3.5.0 plugin set cannot resolve against numpy 1.26.4, and bffile (from
#     bioio-bioformats) needs numpy>=2.1.0. No TensorFlow here, so numpy 2 is safe.
# So this image takes no constraints.txt (the exceptions are recorded in
# test_container_harmonisation's PINNED_EXCEPTIONS); passing it would fail the resolve.
#
# scyjava is named because bin/utils/jvm_cache.py imports it to redirect the jgo cache.
RUN pip3 install --no-cache-dir \
        tifffile==2024.12.12 \
        imagecodecs==2025.3.30 \
        numpy==2.2.6 \
        zarr==2.18.3 \
        numcodecs==0.13.1 \
        h5py==3.11.0 \
        bioio==3.5.0 \
        bioio-ome-tiff==1.4.0 \
        bioio-tifffile==1.3.0 \
        bioio-nd2==2.4.0 \
        bioio-czi==2.9.0 \
        bioio-lif==1.5.0 \
        bioio-bioformats==2.0.0 \
        scyjava==1.12.5 \
        dask[array]==2026.7.1

# Pin the Bio-Formats jar too: unset, jgo resolves whatever upstream released last.
ENV BIOFORMATS_VERSION=ome:formats-gpl:8.1.1

# Bake the Bio-Formats jars and JVM cache. bffile fetches them through scyjava/jgo on first
# use into ~/.jgo and ~/.m2, which fails on the cluster's read-only $HOME and offline.
# Running the reader once with HOME=/root fills /root/.jgo and /root/.m2, which is where
# bin/utils/jvm_cache.py looks first. cjdk also fetches an unused JRE into
# /root/.cache/cjdk during this step.
#
# chmod -R a+rX: /root is mode 700 and tasks run as the host user.
RUN HOME=/root python3 -c "\
import numpy, tifffile; \
tifffile.imwrite('/tmp/bake-probe.tif', numpy.zeros((8, 8), 'uint16')); \
from bioio_bioformats import Reader; \
r = Reader('/tmp/bake-probe.tif'); \
print('baked Bio-Formats reader:', r.dims)" \
    && rm -f /tmp/bake-probe.tif \
    && test -d /root/.m2/repository/ome/formats-gpl \
    && test -n "$(ls -A /root/.m2/repository/ome/formats-gpl)" \
    && test -d /root/.jgo \
    && chmod -R a+rX /root

# The production jvm_cache.py, so smoke.sh tests the same redirect the pipeline uses. Kept
# on its own path so it never shadows the copy the pipeline stages from bin/.
COPY bin/utils/jvm_cache.py /usr/local/lib/mirage_jvm/jvm_cache.py

# Cache directories under /tmp (read-only $HOME). CJDK_CACHE_DIR must stay on the baked
# /root/.cache/cjdk: pointing it at an empty /tmp would send tasks to the network.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba \
    CJDK_CACHE_DIR=/root/.cache/cjdk

# smoke.sh imports what this image's bin/ scripts import, so a broken stack fails the
# build; containers.yml runs it again on the built image. Kept last so the COPY does not
# invalidate earlier layers.
COPY containers/convert/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
