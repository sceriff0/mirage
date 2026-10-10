# DeeperHistReg image for the registration benchmark -- bolt3x/mirage-deeperhistreg:1.0.0
#
# NOT A PIPELINE IMAGE. No modules/local process runs here. It runs one competitor of the
# registration benchmark, `python -m benchmarks.regbench run --method deeperhistreg`
# (benchmarks/regbench/methods/deeperhistreg.py; eight rows, four on a GPU and four on a
# CPU), with the repository bind-mounted. The benchmark's rule is that a method runs as
# its authors publish it, so this is the package's PyPI release 1.0.1 on the stack its
# repository's requirements.txt lists.
#
# ONE IMAGE FOR THE GPU ROWS AND THE CPU ROWS. The base is PyTorch's own CUDA runtime
# image, whose build also runs on a CPU; the device is the adapter's `device` option.
# It is the SAME base and digest as containers/cellsam, for the same reason: 2.6.0 is the
# first release carrying the CVE-2025-32434 (`torch.load`) fix, and it is above this
# benchmark's own floor of 2.3 (uint16 arrays load into torch only from 2.3 on; upstream
# asks for no more than 1.13.1). The framework comes from the base and is not installed
# a second time: requirements/deeperhistreg.txt names the base's two versions so that pip
# leaves them alone, and smoke.sh asserts the CUDA build is still the one imported.
#
# DIGEST-PINNED (ruling R6; tests/test_base_images_are_digest_pinned.py). The digest was
# checked against Docker Hub's registry API on 2026-10-10: the tag still resolves to it.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

# procps: Nextflow's task-metrics wrapper needs `ps`, and although Nextflow does not run
# this image, every image in this directory carries it so the rule has no exception to
# remember (containers/README.md, "Runtime requirements every image must satisfy").
# No OpenSlide or libvips package: `openslide-bin` and `pyvips-binary` are wheels that
# carry those libraries. No display libraries: OpenCV is the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

# The build context is the REPOSITORY ROOT. deeperhistreg.txt opens with
# `-c constraints.txt` and `-r stare.txt`, which pip resolves next to it, so the three are
# copied into one directory. The package declares no dependencies of its own, which is
# why that file lists the whole stack.
COPY requirements/constraints.txt requirements/stare.txt requirements/deeperhistreg.txt /tmp/requirements/
RUN python -m pip install --no-cache-dir -r /tmp/requirements/deeperhistreg.txt

# Read-only $HOME safety, the same three variables the other images set, and BEFORE the
# smoke test here: matplotlib is imported by the package, and the cluster's $HOME is a
# read-only mount.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# The final verification: the same smoke.sh .github/workflows/containers.yml runs with
# `docker run` against the built image, so one definition runs at build time and in CI.
COPY containers/deeperhistreg/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

ENV PYTHONUNBUFFERED=1

CMD ["python", "--version"]

# OCI provenance, last so that VCS_REF invalidates one metadata layer only.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
