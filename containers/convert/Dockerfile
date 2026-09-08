# Bio-Formats reader image for CONVERT_IMAGE — bolt3x/mirage-convert:1.0.0
#
# Build: docker build -t bolt3x/mirage-convert:1.0.0 -f containers/convert/Dockerfile .
#        (from the REPOSITORY ROOT — the build context is `.`)
#
# Image contents:
#   - eclipse-temurin:21-jre-jammy (LTS JRE; JAVA_HOME=/opt/java/openjdk)
#   - python3.10 + tifffile + numpy + h5py
#   - bioio 3.5.0 and SIX reader plugins, including bioio-bioformats
#   - the Bio-Formats jars and JVM cache BAKED under /root (see the bake step below)
#
# THE GLENCOE CLI TOOLS ARE GONE (2026-09-02). bioformats2raw and raw2ometiff were
# installed, symlinked onto $PATH and smoke-tested for their presence -- and invoked by
# NOTHING. modules/local/convert_image.nf runs bin/convert_image.py, whose read path is
# `from bioio import BioImage`. A grep across bin/, modules/, lib/, conf/, workflows/ and
# subworkflows/ found no caller.
#
# BIOIO-BIOFORMATS IS WHAT MAKES THE PIPELINE'S CLAIM TRUE (ruling R2). The other five
# plugins cover OME-TIFF, TIFF, ND2, CZI and LIF; .svs, .qptiff, .vsi, .scn, .mrxs, .bif
# and .ims need the Bio-Formats reader, and bin/utils/ome_io.py's detect_reader() routes
# them to it.
#
# DIGEST-PINNED (ruling R6). The tag stays for readability; the digest is what makes the
# build reproducible -- a version tag is republished, so rebuilding from the tag alone can
# produce a different image than the one that generated the published results. Resolved
# 2026-09-02; Dependabot updates the digest in place. Guarded by
# tests/test_base_images_are_digest_pinned.py.
FROM eclipse-temurin:25-jre-jammy@sha256:10c251954d0bfe1a59ba93505f8c628d755919412400aa98685764c9353605d6

LABEL maintainer="mirage WSI pipeline"
LABEL description="Bio-Formats CONVERT_IMAGE: bioio + bioio-bioformats with baked Bio-Formats jars"

# procps is not optional. Nextflow's task-metrics wrapper hard-exits BEFORE the script
# block when `ps` is absent, and params.enable_trace defaults to true, so every task in
# this image would fail with exit status 1 and empty stdout. Installed in all eleven
# images rather than "verified present in this base" -- apt is a no-op where it already
# is, and containers/<dir>/smoke.sh asserts it either way.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
        coreutils \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first: jammy ships 22.0.2, whose resolver reports this package set's conflict as
# an opaque "nd2[legacy]==0.11.2 and nd2[legacy]==0.11.3" rather than naming the real cap.
RUN pip3 install --no-cache-dir --upgrade pip

# dask is pinned EXPLICITLY even though every bioio package already pulls it in: all seven
# ask only for `dask[array]>=2021.4.1`, with no upper bound, so an unpinned build silently
# takes whatever the newest dask supporting this base's Python 3.10 happens to be on build
# day. bin/convert_image.py's read is dask (BioImage.dask_data) and its write iterates that
# array's CHUNKS, so dask's version is load-bearing for this image, not incidental.
# 2026.7.1 is the version containers/stardist and containers/cellsam already pin, and the
# one CI pins.
#
# numpy and tifffile diverge from the harmonised set here, and both are forced:
#   * bioio-ome-tiff 1.4.0 requires tifffile[zarr]<2025.1.10 on python_version < "3.11", and this
#     base (eclipse-temurin:21-jre-jammy) is Python 3.10. 2024.12.12 is the newest that satisfies it.
#   * the bioio 3.5.0 plugin set cannot resolve against numpy 1.26.4 -- bioio-lif's dask chain is
#     unsatisfiable there. numpy 2 is safe in THIS image specifically: it carries no TensorFlow or
#     StarDist, which are the only reason the harmonised numpy is held at the last 1.x.
# numpy 2 is also what bioio-bioformats needs: it depends on bffile 0.1.1, which requires
# numpy>=2.1.0. The existing exception covers it; no new one was needed.
#
# THIS IMAGE TAKES NO requirements/constraints.txt, and that is deliberate -- it is the
# second documented exception alongside containers/spatialdata. Both divergences above are
# already recorded in tests/test_container_harmonisation.py's PINNED_EXCEPTIONS, each with
# the upstream constraint that forces it, and tests/test_ci_stack_pinned.py derives its
# exception list from that table rather than from a hand-written name list. Passing
# `-c /tmp/requirements/constraints.txt` here would not "let the constraint win" -- pip
# would report ResolutionImpossible on numpy and tifffile and the image would not build.
#
# scyjava is named explicitly even though bffile pulls it in: it is what
# bin/utils/jvm_cache.py's point_jvm_cache_off_readonly_home() imports
# (scyjava.config.set_cache_dir/set_m2_repo -- the only timing-immune way to redirect
# the jgo cache), which both bin/utils/ome_io.py::_open_bioio and
# bin/convert_image.py::read_image call for the bioio-bioformats route, and which
# smoke.sh below also calls via the same production function. An install nothing in
# this Dockerfile names is invisible to test_container_harmonisation.py's
# "installs what it imports" guard.
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

# PIN THE JAVA SIDE TOO (ruling R6 applies to more than pip). bffile reads
# BIOFORMATS_VERSION and accepts a full Maven coordinate; unset, the Bio-Formats jar jgo
# resolves changes whenever upstream releases. 8.1.1 is a released version in the OME
# artifactory's ome/formats-gpl metadata and the form bffile's own documentation uses.
ENV BIOFORMATS_VERSION=ome:formats-gpl:8.1.1

# BAKE THE BIO-FORMATS JARS AND THE JVM CACHE. bffile fetches ome:formats-gpl through
# scyjava/jgo at FIRST USE into ~/.jgo + ~/.m2. The cluster has a READ-ONLY $HOME and
# docs/usage.md documents air-gapped execution, so a first run there dies exactly the
# way bin/utils/jvm_cache.py's docstring records: jgo's os.makedirs($HOME/.jgo) raising
# OSError [Errno 30] Read-only file system.
#
# CJDK DOES FIRE DURING THIS BAKE, MEASURED, DESPITE THE BASE ALREADY HAVING A JRE.
# JAVA_HOME=/opt/java/openjdk is set (see the header above), and scyjava's JVM discovery
# finds and uses it directly at runtime -- but Task 13's real build observed cjdk fetching
# a zulu-jre into ~/.cache/cjdk during the bake step below anyway. That fetched JRE is dead
# weight at runtime (the base image's own JDK is what actually runs), but it still sits
# under $HOME (this bake's HOME=/root, so /root/.cache/cjdk), and is covered like
# /root/.jgo and /root/.m2 by the `chmod -R a+rX /root` below.
#
# HOME=/root is the whole mechanism, and it needs no configuration API: scyjava computes
# its jgo cache_dir and m2_repo from pathlib.Path.home() at import time, so running the
# reader once as root populates /root/.jgo and /root/.m2. That is precisely where
# bin/utils/jvm_cache.point_jvm_cache_off_readonly_home() looks first -- "prefer a warm
# baked cache (/root/.jgo + /root/.m2): jgo short-circuits on an existing buildSuccess
# workspace, so pointing scyjava here needs no writes and no network".
#
# chmod -R a+rX is not optional: /root is mode 700 and nextflow.config's docker profile
# runs tasks with `-u $(id -u):$(id -g)`.
#
# This is a CONTAINER-ONLY fix. A git pull can never repair a missing jar.
# Guarded by tests/test_bioformats_jars_are_baked.py.
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

# COPY the PRODUCTION jvm-cache redirect so smoke.sh calls the exact same function the
# real pipeline calls, instead of hand-rolling its own scyjava.config.set_cache_dir/
# set_m2_repo pair. A hand-rolled pair only proves scyjava CAN be redirected -- it does
# not prove anything on the CONVERT_IMAGE run path actually redirects it, which is what
# review round 1 on Task 7 found missing (bin/utils/ome_io.py::_open_bioio and
# bin/convert_image.py::read_image now both call
# jvm_cache.point_jvm_cache_off_readonly_home() for the bioio-bioformats route). Placed
# on its own PYTHONPATH entry, not merged into the image's site-packages, so it is
# obviously a smoke-test-only copy and never shadows the real bin/utils/jvm_cache.py the
# pipeline stages fresh from the git checkout on every run.
COPY bin/utils/jvm_cache.py /usr/local/lib/mirage_jvm/jvm_cache.py

# Fail the BUILD rather than a cluster run. containers/<dir>/smoke.sh imports exactly
# the third-party modules this image's bin/ scripts import at run time, so a stack that
# resolved in pip and then explodes on first import stops here. The SAME script is run
# again by .github/workflows/containers.yml against the built image, so there is one
# definition exercised in two places rather than two copies that drift.
#
# LAST, and after every pip install, on purpose: a COPY invalidates the layer cache for
# everything below it, so putting this at the end costs no rebuilds.
# Read-only $HOME safety, as in containers/segeval and containers/spatialdata.
#
# CJDK_CACHE_DIR IS PINNED SEPARATELY AND MUST NOT FOLLOW XDG_CACHE_HOME. The Java runtime
# scyjava may fetch through cjdk was baked under /root/.cache/cjdk by the bake step above
# and made world-readable; redirecting the lookup to an empty /tmp would send a task to the
# network, which is the exact failure the bake exists to prevent. The jgo and Maven caches
# are not XDG directories at all -- bin/utils/jvm_cache.py points scyjava at /root/.jgo and
# /root/.m2 through its config API.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba \
    CJDK_CACHE_DIR=/root/.cache/cjdk

COPY containers/convert/smoke.sh /usr/local/bin/mirage-smoke.sh
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
