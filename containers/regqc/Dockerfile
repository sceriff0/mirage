# Registration-QC image — bolt3x/mirage-regqc:1.0.0
#
# One process: GENERATE_REGISTRATION_QC (bin/generate_registration_qc.py -> bin/utils/qc.py),
# which imports cv2, numpy, tifffile, skimage.transform and zarr only; smoke.sh asserts that set.
#
# ffmpeg / libsm6 / libxext6 are the runtime libraries of the non-headless opencv-python.
#
# Nothing here uses CUDA; the CUDA base is kept because swapping it would also change
# opencv's system libraries.
#
# Digest-pinned for reproducibility (tags get republished); Dependabot updates it.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04@sha256:217134b60289ced62b47463be32584329baf35cb55a9ccda6626b91bd59803be

ENV DEBIAN_FRONTEND=noninteractive

# procps: Nextflow's task-metrics wrapper needs `ps`, or every task exits 1 before its script.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    procps \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Build context is the repository root. regqc.txt's `-c constraints.txt` resolves next to
# it, so both land in one directory. A pin that drifts from constraints.txt fails the build.
COPY requirements/constraints.txt requirements/regqc.txt /tmp/requirements/

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# `python3 -m pip` so pins land in the interpreter the task runs.
# scipy is pinned in requirements/regqc.txt, the last pip step, so no later step can move it.
RUN python3 -m pip install --no-cache-dir -c /tmp/requirements/constraints.txt \
    numpy==1.26.4 \
    scikit-image==0.25.2

RUN python3 -m pip install --no-cache-dir -r /tmp/requirements/regqc.txt

# Cache directories under /tmp: tasks run as the host user and the cluster's $HOME is
# read-only, so a library writing to ~/.cache or ~/.config would crash.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# smoke.sh imports what this image's bin/ scripts import, so a broken stack fails the
# build; containers.yml runs it again on the built image. Kept last so the COPY does not
# invalidate earlier layers.
COPY containers/regqc/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# OCI provenance, last because VCS_REF changes every commit and would invalidate later
# layers. containers.yml passes both values; a local build gets "unknown".
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
LABEL org.opencontainers.image.source="https://github.com/sceriff0/mirage" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
