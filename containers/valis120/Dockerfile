# VALIS 1.2.0 image for the registration benchmark -- bolt3x/mirage-valis120:1.0.0
#
# NOT A PIPELINE IMAGE. No modules/local process runs here: the pipeline's VALIS is
# upstream's 1.0.0 image (modules/local/register.nf), which the benchmark's `valis_mirage`
# row also uses. This image runs the benchmark's CURRENT-RELEASE rows,
# `python -m benchmarks.regbench run --method valis --label valis120` and
# `--label valis120_hires` (benchmarks/regbench/methods/valis.py), with the repository
# bind-mounted.
#
# UPSTREAM'S OWN IMAGE, NOT A REBUILD OF IT (ruling R3.9b, 2026-10-10). The benchmark's
# rule is that a method runs as its authors publish it, and VALIS's authors publish an
# image. Everything VALIS runs on here is what they built: nothing below reinstalls VALIS,
# torch, Kornia, pyvips, libvips, the JVM or Bio-Formats.
#
# WHAT UPSTREAM'S IMAGE CONTAINS, AND HOW THAT WAS LEARNED. No Docker daemon ran where
# this file was written, so nothing here comes from a running container. It was read on
# 2026-10-11 from Docker Hub's registry API (registry-1.docker.io, anonymous pull token):
# the index of the tag, the linux/amd64 manifest, its config blob, and its layers'
# file lists, streamed and listed without unpacking.
#   index (what the tag points at; pinned below)
#       sha256:4fd20b687ae0f08c47011753bc5083dca5ac65a454d3615589eddf6b14e3c419
#   linux/amd64 manifest   sha256:2797af6d99fa75032a5859f65c67b64290b087bfb291d2d66d390325d06f328b
#   its config             sha256:89273efc80887ac1f9eb81c33181d9c69bb9d44634f09f0841421895fec2cedc
#   created 2025-06-25; 26 layers, 5.56 GB compressed (3.49 GB of it one layer: the
#   Python stack with CUDA libraries; 1.35 GB the weights)
#   [config, history]  Ubuntu 24.04. apt: openjdk-11-jre (JVM 11 at
#       /usr/lib/jvm/java-11-openjdk-amd64), python3-venv. libvips 8.16.0 built from
#       source into /usr/local/lib (LD_LIBRARY_PATH=/usr/local/lib). Bio-Formats 7.0.0:
#       bioformats_package.jar fetched into /usr/local/src/valis. WORKDIR /usr/local/src,
#       CMD /bin/bash, PATH=/usr/local/src/.venv/bin:/root/.local/bin/:... (the first
#       entry does not exist in any layer listed).
#   [layer 5, pyvenv.cfg]  Python 3.12.3, a virtual environment at /root/.local.
#   [layer 22, dist-info names]  91 distributions in /root/.local/lib/python3.12/
#       site-packages, among them: valis-wsi 1.2.0 (installed from /usr/local/src by uv),
#       torch 2.7.1 with the nvidia-*-cu12 12.6 wheels (the CUDA build), torchvision
#       0.22.1, kornia 0.8.1, pyvips 3.0.0, numpy 1.26.4, scipy 1.16.0, scikit-image
#       0.25.2, pandas 2.3.0, tifffile 2025.6.11, opencv-contrib-python-headless
#       4.9.0.80, jpype1 1.5.2, scyjava 1.12.1, simpleitk 2.5.2, psutil 7.0.0.
#       NOT there: zarr, numcodecs, imagecodecs, pytest, stare-registration.
#   [layer 25]  six weight files in /root/.cache/torch/hub/checkpoints (DISK, the DISK
#       and DeDoDe LightGlue weights, DeDoDe's detector and descriptor, DINOv2 ViT-L/14).
#   [layers 1, 3, 5, 8, 21, 23, 25]  /root is mode 0700 in every one.
#
# WHAT THAT MEANS ON A CLUSTER, AND WHAT THIS FILE DOES ABOUT IT. Upstream's image is
# built to be run as root by Docker. Apptainer runs it as the calling user, with the image
# read-only and another $HOME, and then four things break. Each is put right here with
# upstream's own files (containers/valis120/place_assets.py has the reading behind 2 to 4):
#   1. NO PYTHON. The interpreter and every package are under /root, which only root can
#      enter. `chmod a+rx /root` below.
#   2. NO BIO-FORMATS JAR beside the installed package (it is beside the source tree,
#      which is what `import valis` finds only from /usr/local/src), so VALIS would ask
#      Maven for it. The jar is linked beside the installed package.
#   3. NO WEIGHTS. They are in root's torch cache; another user's torch.hub looks in that
#      user's own home and downloads on `import valis`. TORCH_HOME=/opt/torch is set and
#      linked to upstream's directory.
#   4. <package>/data/bf_formats.txt, which VALIS 1.2.0 REWRITES on every JVM start, is
#      root's. It is made writable by any user. THAT IS NOT ENOUGH UNDER APPTAINER: the
#      image is mounted read-only there, so the run must add `--writable-tmpfs` (or the
#      JVM start raises "Read-only file system"). Nothing in this file can do that for
#      the caller.
# smoke.sh then imports VALIS and starts its JVM as an unprivileged user, from another
# directory, with another $HOME and with the network closed.
#
# WHAT THE BENCHMARK ADDS (requirements/valis120.txt): zarr 2, numcodecs, imagecodecs
# and pytest at the repository's harmonised versions, and STARE's reader (the one pin of
# requirements/stare.txt, installed with --no-deps). psutil is already upstream's. The
# install is constrained twice: by requirements/constraints.txt, and by a freeze of the
# base image, so it can add and cannot move. The ONE version of upstream's stack that
# changes is tifffile (2025.6.11 -> 2025.5.10), which STARE's reader needs and VALIS does
# not import. valis120.txt says why at length, and smoke.sh fails the build if any other
# line of upstream's freeze is different afterwards.
#
# THE TORCH HERE IS UPSTREAM'S CUDA BUILD. Neither VALIS row asks for a GPU and the
# benchmark gives them none, so it runs on the CPU; it would use a GPU if one were made
# visible to the container.
#
# DIGEST-PINNED (ruling R6; tests/test_base_images_are_digest_pinned.py). The digest is
# the multi-platform index Docker Hub served for the tag on 2026-10-11 (above); the tag
# stays for readability.
FROM cdgatenbee/valis-wsi:1.2.0@sha256:4fd20b687ae0f08c47011753bc5083dca5ac65a454d3615589eddf6b14e3c419

# procps: Nextflow's task-metrics wrapper needs `ps`. Ubuntu 24.04's base layer already
# has /usr/bin/ps (read from layer 1), so this installs nothing or a point update; it is
# written out because every image in this directory names it, so the rule has no
# exception to remember (containers/README.md, "Runtime requirements every image must
# satisfy").
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Point 1 above. /root holds the Python environment (/root/.local) and the weights
# (/root/.cache/torch); below it upstream's directories are already 0755. Nothing private
# is in there: it is a published image.
RUN chmod a+rx /root

# The build context is the REPOSITORY ROOT. valis120.txt opens with `-c constraints.txt`,
# which pip resolves next to it. requirements/stare.txt is copied for ONE line: STARE's
# pin, cut out so that its version is still written exactly once in this repository
# (`test -s` fails the build if stare.txt no longer has it). The rest of stare.txt is not
# used: it would re-pin upstream's scipy.
COPY requirements/constraints.txt requirements/stare.txt requirements/valis120.txt /tmp/requirements/

# The freeze of upstream's stack, taken BEFORE anything is added, and kept in the image
# (/opt/mirage/upstream-freeze.txt): it is the second constraint of the install and what
# smoke.sh compares the final stack with. tifffile is left out of the constraint because
# it is the one version this image changes. `pip` here is the virtual environment's own
# (`python -m pip`); it is not upgraded.
RUN mkdir -p /opt/mirage \
    && python -m pip freeze --all | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' | LC_ALL=C sort > /opt/mirage/upstream-freeze.txt \
    && test "$(wc -l < /opt/mirage/upstream-freeze.txt)" -ge 85 \
    && grep -q '^tifffile==2025.6.11$' /opt/mirage/upstream-freeze.txt \
    && grep -v -i '^tifffile==' /opt/mirage/upstream-freeze.txt > /tmp/requirements/upstream-constraints.txt \
    && chmod -R a+rX /opt/mirage
RUN python -m pip install --no-cache-dir \
        -c /tmp/requirements/upstream-constraints.txt \
        -r /tmp/requirements/valis120.txt

# STARE, WITHOUT ITS DEPENDENCIES (requirements/valis120.txt says why): its requirements
# are met by what is installed, and smoke.sh proves that from its own metadata.
RUN grep -E '^stare-registration @ ' /tmp/requirements/stare.txt > /tmp/requirements/stare-only.txt \
    && test -s /tmp/requirements/stare-only.txt \
    && python -m pip install --no-cache-dir --no-deps -r /tmp/requirements/stare-only.txt

# Where torch.hub looks for weights, for EVERY user of the image (point 3). Set before
# place_assets.py (which refuses to run without it) and before the smoke test. The other
# three are the read-only-$HOME variables every image here sets; matplotlib is imported
# by VALIS.
ENV TORCH_HOME=/opt/torch \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# Points 2 to 4: the jar, the weights and the permissions of <package>/data. No network.
COPY containers/valis120/place_assets.py /usr/local/bin/mirage-valis120-place-assets.py
RUN python /usr/local/bin/mirage-valis120-place-assets.py

# The final verification, and here it is the proof of points 1 to 4: smoke.sh imports
# VALIS and starts its JVM as an unprivileged user with the network closed. The same
# script runs again through `docker run` in .github/workflows/containers.yml.
COPY containers/valis120/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

ENV PYTHONUNBUFFERED=1

# Upstream's CMD is /bin/bash; this is the one every image here ends with.
CMD ["python", "--version"]

# OCI provenance, last so that VCS_REF invalidates one metadata layer only.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
