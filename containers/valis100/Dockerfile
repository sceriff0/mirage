# VALIS 1.0.0 image for the registration benchmark -- bolt3x/mirage-valis100:1.0.0
#
# NOT A PIPELINE IMAGE. The pipeline's REGISTER runs upstream's image itself
# (modules/local/register.nf). This is that SAME image, by the same digest, for the
# benchmark's `valis_mirage` row (benchmarks/regbench/methods/valis_mirage.py), with the
# repository bind-mounted.
#
# UPSTREAM'S OWN IMAGE, NOT A REBUILD OF IT. Nothing below reinstalls VALIS or anything
# it runs on. Added (requirements/valis100.txt): zarr 2, numcodecs and imagecodecs, which
# the harness's slide reader needs and upstream's image does not have -- the benchmark's
# preflight refused the bare image on 2026-10-13. The install is constrained by a freeze
# of the base, and smoke.sh fails the build if any line of that freeze is different
# afterwards.
#
# WHAT WAS NOT CHECKED. No Docker daemon ran where this file was written: the base's
# Python, numpy and tifffile versions were not read. The build is where the pins are
# first shown to install on them, and smoke.sh where they are first shown to work.
#
# DIGEST-PINNED: the digest modules/local/register.nf runs.
FROM cdgatenbee/valis-wsi:1.0.0@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6

# procps: Nextflow's task-metrics wrapper needs `ps`; every image here names it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

# The build context is the REPOSITORY ROOT. requirements/stare.txt is copied for ONE line,
# STARE's pin, so that its version is still written exactly once in this repository.
COPY requirements/constraints.txt requirements/stare.txt requirements/valis100.txt /tmp/requirements/

# The freeze of upstream's stack, taken BEFORE anything is added and kept in the image
# (/opt/mirage/upstream-freeze.txt): the first constraint of the install, and what
# smoke.sh compares the final stack with.
#
# UPSTREAM WINS. The base's versions were not read where this was written, so a line of
# requirements/constraints.txt or of requirements/valis100.txt that names a distribution
# the base ALREADY HOLDS is dropped before the install (the awk below, on normalised
# names): the harmonised pins apply to what is added and never ask upstream's stack to
# move. /opt/mirage/dropped-pins.txt keeps what was dropped, and the build prints it.
RUN PY="$(command -v python || command -v python3)" \
    && mkdir -p /opt/mirage \
    && "$PY" -m pip freeze --all | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' | LC_ALL=C sort > /opt/mirage/upstream-freeze.txt \
    && test -s /opt/mirage/upstream-freeze.txt \
    && for f in constraints valis100; do \
         awk -F'==' -v dropped=/opt/mirage/dropped-pins.txt \
           'NR==FNR { n=tolower($1); gsub(/[._]/,"-",n); held[n]=1; next } \
            /^[A-Za-z0-9]/ { n=tolower($1); gsub(/[._]/,"-",n); if (n in held) { print FILENAME ": " $0 >> dropped; next } } \
            { print }' /opt/mirage/upstream-freeze.txt "/tmp/requirements/$f.txt" > "/tmp/requirements/$f.kept" \
         && mv "/tmp/requirements/$f.kept" "/tmp/requirements/$f.txt"; \
       done \
    && touch /opt/mirage/dropped-pins.txt && echo "pins dropped because upstream holds them:" && cat /opt/mirage/dropped-pins.txt \
    && chmod -R a+rX /opt/mirage \
    && "$PY" -m pip install --no-cache-dir \
        -c /opt/mirage/upstream-freeze.txt \
        -r /tmp/requirements/valis100.txt

# STARE, WITHOUT ITS DEPENDENCIES: the harness reads every slide through stare.slide_io,
# whose imports (numpy, tifffile, zarr) are met by what is installed; smoke.sh proves it
# by reading a file through it. scipy and scikit-image, which the rest of STARE asks for,
# are upstream's or absent, and nothing this image runs reaches them.
RUN PY="$(command -v python || command -v python3)" \
    && grep -E '^stare-registration @ ' /tmp/requirements/stare.txt > /tmp/requirements/stare-only.txt \
    && test -s /tmp/requirements/stare-only.txt \
    && "$PY" -m pip install --no-cache-dir --no-deps -r /tmp/requirements/stare-only.txt

# The read-only-$HOME variables every image here sets.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# The same script runs again through `docker run` in .github/workflows/containers.yml.
COPY containers/valis100/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last so that VCS_REF invalidates one metadata layer only.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
