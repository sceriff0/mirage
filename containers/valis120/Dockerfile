# VALIS 1.2.0 image for the registration benchmark -- bolt3x/mirage-valis120:1.0.0
#
# NOT A PIPELINE IMAGE. No modules/local process runs here: the pipeline's VALIS is
# upstream's 1.0.0 image (modules/local/register.nf), which the benchmark's `valis_mirage`
# row also uses. This image runs the benchmark's CURRENT-RELEASE rows,
# `python -m benchmarks.regbench run --method valis --label valis120` and
# `--label valis120_hires` (benchmarks/regbench/methods/valis.py), with the repository
# bind-mounted. The rule is that a method runs as its authors publish it: this is the PyPI
# release `valis-wsi==1.2.0` on the stack its metadata asks for.
#
# WHAT A PIP INSTALL OF VALIS LEAVES TO THE FIRST RUN, AND THIS IMAGE DOES AT BUILD TIME.
# A compute node may have no network and its $HOME is read-only, so each of these would
# otherwise fail far from its cause. containers/valis120/fetch_assets.py has the reading
# behind every line here.
#   1. Bio-Formats. VALIS starts its JVM from <package>/bioformats_package.jar when the
#      file is there, and asks Maven for it when it is not. The jar is fetched here, at
#      the version upstream's own image uses (7.0.0), against OME's published sha256.
#   2. The DISK and LightGlue weights. `import valis` builds the default matcher, which
#      loads both through torch.hub: a download into ~/.cache on first import. They are
#      fetched here into TORCH_HOME=/opt/torch, sha256-checked, and smoke.sh then imports
#      VALIS with every proxy variable pointed at a closed port.
#   3. <package>/data/bf_formats.txt, which VALIS 1.2.0 REWRITES on every JVM start.
#      It is made writable by any user. THAT IS NOT ENOUGH UNDER APPTAINER: the image is
#      mounted read-only there, so the run must add `--writable-tmpfs` (or the JVM start
#      raises "Read-only file system"). Nothing in this file can do that for the caller.
#
# CPU ONLY. Neither VALIS row asks for a GPU, so torch is the CPU wheel from PyTorch's
# own index (requirements/valis120.txt); the default PyPI wheel would add about 2 GB of
# CUDA libraries that nothing here loads.
#
# NOT `FROM cdgatenbee/valis-wsi:1.2.0`, although upstream publishes that image. Recorded
# because it is the obvious alternative: that image is 5.6 GB compressed, builds libvips
# from source and installs whatever torch resolved on its build day; it sets no
# TORCH_HOME (so its build-time weight download went to the build user's cache) and puts
# its jar beside the SOURCE tree it then installs a copy of (both read from its build
# history on 2026-10-10, not from a running container), so points 1 to 3 would need the
# same work there. The benchmark's plan names the pip release; this follows it.
#
# DIGEST-PINNED (ruling R6; tests/test_base_images_are_digest_pinned.py): the SAME base
# and digest as containers/stare (Python 3.11.16 on Debian trixie).
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# procps: Nextflow's task-metrics wrapper needs `ps`, and although Nextflow does not run
# this image, every image in this directory carries it so the rule has no exception to
# remember. default-jre-headless: the JVM that JPype starts for Bio-Formats (OpenJDK 21 on
# trixie; upstream's image has OpenJDK 11, and Bio-Formats 7 runs on either). The man
# directory is created first because a JRE's post-install script links into it and slim
# images ship without it. libvips needs no package: it comes inside the pyvips-binary
# wheel.
RUN mkdir -p /usr/share/man/man1 \
    && apt-get update && apt-get install -y --no-install-recommends \
        procps \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

RUN pip install --no-cache-dir --upgrade pip

# The build context is the REPOSITORY ROOT. valis120.txt opens with `-c constraints.txt`
# and `-r stare.txt`, which pip resolves next to it, so the three are copied into one
# directory. It pins every distribution the resolver installs.
COPY requirements/constraints.txt requirements/stare.txt requirements/valis120.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/valis120.txt

# Where torch.hub looks for weights, for EVERY user of the image: without it the cache is
# ~/.cache/torch of whoever is running, which at build time is root and on the cluster
# is not. Set before the fetch (which refuses to run without it) and before the smoke
# test. The other three are the read-only-$HOME variables every image here sets;
# matplotlib is imported by VALIS.
ENV TORCH_HOME=/opt/torch \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# The jar, the two weight files, and the permissions of <package>/data (points 1 to 3
# above). The only step of this build that needs a network besides pip and apt.
COPY containers/valis120/fetch_assets.py /usr/local/bin/mirage-valis120-fetch-assets.py
RUN python /usr/local/bin/mirage-valis120-fetch-assets.py

# The final verification, and here it is the proof of points 1 and 2: smoke.sh imports
# VALIS and starts its JVM with the network closed. The same script runs again through
# `docker run` in .github/workflows/containers.yml.
COPY containers/valis120/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

ENV PYTHONUNBUFFERED=1

CMD ["python", "--version"]

# OCI provenance, last so that VCS_REF invalidates one metadata layer only.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
