# STARE tiled registration image — bolt3x/mirage-tiled:1.0.0
#
# Runs the fully-parallel, JVM-free registration method (registration_method='tiled'):
# TILED_COARSE, TILED_REG_TILE, TILED_SOLVE, TILED_STITCH and WARP_SEG_QC's tiled backend
# (lib/WarpBackends.groovy). The method is the `stare` package (packages/stare), installed
# below; bin/tiled_*.py are shims over its stages. (The single-task TILED_REGISTER this comment used to name was
# removed with the reg_tiled_fanout flag — see CHANGELOG.)
# NO Java/BioFormats, NO libvips, NO GPU — that is what keeps this image an order of
# magnitude smaller than the VALIS image. It no longer makes the METHOD laptop-sized:
# TILED_COARSE's DISK matcher asks ~48 GB at the shipped reg_tiled_mode=high tier
# (~5 GB at low), so STARE is a cluster backend at its default.
#
# THIS IMAGE NOW CARRIES torch + kornia (CPU wheels), and the earlier "lean, pure-CPU
# scientific-Python, torch-free" claim is deliberately RETIRED. They power the DISK +
# LightGlue COARSE front-end (stare/coarse_align.py::_frontend_disk_lightglue), which
# until now lived in a second, optional image (containers/stare-ml, `-profile stare_ml`).
# That split is gone: ONE image that always works beats TWO where the second is never
# published. The second image was built by a plain push to main but NEVER pushed
# (.github/workflows/containers.yml pushes only on release:published /
# workflow_dispatch), so `-profile stare_ml` failed on image pull for anyone who had not
# manually dispatched the build — the exact trap `:tiled` itself shipped into once
# already. Folding torch+kornia in here costs every user ~1 GB of pull; it buys every user
# a front-end that runs. The image still has no JVM, no libvips and no CUDA runtime.
#
# CPU-ONLY, deliberately: torch comes from `--index-url https://download.pytorch.org/whl/cpu`
# because the cluster this pipeline ships to never guarantees a GPU-visible node for
# TILED_COARSE, and the CPU path must stay correct since the default deployment exercises
# nothing else. `torch.cuda.is_available()` must report False out of this image; True means
# the wrong (CUDA) wheel got installed, and the final verification RUN below asserts it.
# DIGEST-PINNED (ruling R6). The tag stays for readability; the digest is what makes the
# build reproducible -- a version tag is republished, so rebuilding from the tag alone can
# produce a different image than the one that generated the published results. Resolved
# 2026-09-02; Dependabot updates the digest in place. Guarded by
# tests/test_base_images_are_digest_pinned.py.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# scikit-image / scipy import cleanly on slim with no extra *Python-facing* system libs, but
# `procps` is not optional: Nextflow's task-metrics wrapper hard-exits BEFORE the script block
# when `ps` is absent -- nextflow/executor/command-trace.txt does
#     command -v ps &>/dev/null || { >&2 echo "Command 'ps' required by nextflow ..."; exit 1; }
# so every task in this image failed with exit status 1 and empty stdout as soon as the run
# collected metrics (-with-report / -with-trace, i.e. any run producing the resource report).
# python:*-slim ships no procps. Same reason it is installed in containers/spatialdata/Dockerfile.
#
# libgl1 / libglib2.0-0: torch/kornia's transitive OpenCV-adjacent wheels dlopen libGL at
# import time even in a headless container; without them `import kornia` fails at RUNTIME,
# not at build time, which is a worse place to discover it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# The build context is the REPOSITORY ROOT (containers.yml passes `context: .`), so the
# pin files come from requirements/ -- the same files .github/workflows/*.yml install. There
# is no second copy of any version string to keep in step.
COPY requirements/constraints.txt requirements/tiled.txt requirements/torch-cpu.txt requirements/kornia.txt /tmp/requirements/
# torch BEFORE kornia, and from the CPU index: kornia pulls torch from PyPI (the CUDA
# wheel, ~2.5 GB) if torch is not already satisfied. The order is load-bearing, and it is
# why those two are separate files and separate invocations -- see requirements/torch-cpu.txt.
# tiled.txt opens with `-c constraints.txt`, which pip resolves next to it.
RUN pip install --no-cache-dir -r /tmp/requirements/tiled.txt \
    && pip install --no-cache-dir -r /tmp/requirements/torch-cpu.txt \
    && pip install --no-cache-dir -r /tmp/requirements/kornia.txt

# THE METHOD ITSELF. STARE's source of truth is the `stare` package under packages/stare
# (its stages, the solve, the streaming stitch); the bin/tiled_*.py scripts the modules
# invoke are shims over it, so the image must carry it or every TILED_* task fails at
# import. Installed AFTER the requirements: its pyproject dependencies are all already
# satisfied at the constrained versions, so this step resolves nothing new (and must not --
# a version it pulled on its own would be a fourth copy of a pin). The build context is the
# repository root, which is why `packages/stare` is COPY-able at all.
# Guarded by tests/test_container_harmonisation.py's first-party-package check and by
# tests/test_container_smoke_tests.py (smoke.sh imports it).
COPY packages/stare /tmp/stare
RUN pip install --no-cache-dir --no-deps /tmp/stare \
    && rm -rf /tmp/stare

# BAKE THE PRETRAINED WEIGHTS. kornia fetches these at FIRST USE into
# $TORCH_HOME/hub/checkpoints -- from raw.githubusercontent.com (DISK) and
# github.com/cvg/LightGlue/releases (LightGlue). The cluster has a READ-ONLY $HOME and
# docs/usage.md documents air-gapped execution, so a first run would die there with no
# network and nowhere to write. TORCH_HOME is set to /opt/torch, outside $HOME, and made
# world-readable so any task UID can read it (docker.runOptions passes -u $(id -u)).
#
# The two filenames are the ones torch.hub derives from the download URLs, so the LightGlue
# checkpoint is `disk_lightglue_v0-1_arxiv-pth` -- no dot before `pth`. That is not a typo;
# do not "correct" it.
#
# This is a CONTAINER-ONLY fix. A git pull can never repair a missing checkpoint.
# Guarded by tests/test_disk_weights_are_baked.py.
ENV TORCH_HOME=/opt/torch
RUN python -c 'from kornia.feature import DISK, LightGlue; DISK.from_pretrained("depth"); LightGlue("disk")' \
    && test -f /opt/torch/hub/checkpoints/depth-save.pth \
    && test -f /opt/torch/hub/checkpoints/disk_lightglue_v0-1_arxiv-pth \
    && chmod -R a+rX /opt/torch \
    && HOME=/nonexistent python -c 'from kornia.feature import DISK, LightGlue; DISK.from_pretrained("depth"); LightGlue("disk"); print("baked weights load with no HOME")'

# The final verification. IT MOVED, VERBATIM, into containers/tiled/smoke.sh (CI
# redesign Phase 6, 2026-09-01) -- including the CPU-wheel assertion above -- so the
# same definition runs here at build time AND from .github/workflows/containers.yml
# with `docker run` against the built image.
COPY containers/tiled/smoke.sh /usr/local/bin/mirage-smoke.sh
RUN bash /usr/local/bin/mirage-smoke.sh

# Carried over from the folded-in stare-ml image, which set it: TILED_COARSE used to run
# there under -profile stare_ml and now runs here, so keeping it preserves that path's
# behaviour. Unbuffered stdout also means a long COARSE task's progress lines reach
# Nextflow's .command.out as they happen rather than at exit.
ENV PYTHONUNBUFFERED=1

# Read-only $HOME safety, the same three variables containers/segeval and
# containers/spatialdata set. Several of these libraries default their caches to ~/.cache;
# nextflow.config's docker profile runs tasks with `-u $(id -u):$(id -g)` and the cluster's
# $HOME is a read-only mount, so a cache write turns into a crash far from its cause -- the
# same class of failure as the VALIS JVM/jgo cache. MPLCONFIGDIR is not decoration here:
# matplotlib writes ~/.config/matplotlib on its first import.
ENV XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig \
    NUMBA_CACHE_DIR=/tmp/.numba

# The bin/ scripts also redirect numba/matplotlib themselves; these ENV lines make the
# image correct for a script that forgets to, and match containers/segeval and
# containers/spatialdata. TORCH_HOME above is separate and must NOT follow XDG_CACHE_HOME:
# the DISK/LightGlue checkpoints are baked under /opt/torch.
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
