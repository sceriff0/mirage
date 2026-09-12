#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-tiled.
#
# ONE DEFINITION, RUN IN TWO PLACES: this image's own Dockerfile RUNs it as a build
# step, and .github/workflows/containers.yml runs it again with `docker run` against
# the built image on a pull request that touched this build context. A stack that
# resolves in pip and then explodes on first import therefore fails the BUILD, not a
# cluster task six hours in -- scikit-image 0.25 against numpy 1.26 is a pairing pip
# will happily accept and then break on.
#
# WHAT IT ASSERTS, AND WHY THAT SET. Exactly the third-party modules the pipeline's
# own bin/ scripts import when they run in THIS image -- bin/tiled_coarse.py, tiled_reg_tile.py,
# tiled_solve.py and tiled_stitch.py, i.e. the `stare` package they shim over, plus the
# torch/kornia DISK+LightGlue COARSE front-end in stare/coarse_align.py. MOVED HERE VERBATIM
# from this image's Dockerfile on 2026-09-01 (the CPU-wheel assertion included); the
# `stare` import block was added when the method moved into packages/stare.
# That rule is what makes the list neither a guess nor decoration: every name below
# is imported on a live run, so a name that stops importing is a broken image by
# definition. Do not add a module this image's processes never import (an assertion
# nothing depends on fails the build for no user-visible reason); do not drop one
# they do.
set -euo pipefail

# `python` is NOT present in every base used here -- containers/convert installs
# python3 with no `python` alternative -- so resolve the interpreter rather than
# assuming a name.
PY="$(command -v python || command -v python3)"

"$PY" -c "import numpy, scipy, skimage, tifffile, zarr, torch, kornia; \
assert not torch.cuda.is_available(), 'CUDA wheel installed -- expected the CPU wheel'; \
print('tiled image OK:', numpy.__version__, scipy.__version__, skimage.__version__, tifffile.__version__, zarr.__version__, torch.__version__, kornia.__version__)"

# The method itself: the `stare` package (packages/stare, pip-installed by the Dockerfile),
# which bin/tiled_*.py shim over. A stage module that stops importing is a broken image
# just as a missing wheel is; importing all four proves the package AND its stage graph.
"$PY" -c "import stare, stare.stages.coarse, stare.stages.reg_tile, stare.stages.solve, stare.stages.stitch; \
print('stare package OK:', stare.__version__)"

# procps supplies `ps`; Nextflow's task-metrics wrapper hard-exits without it.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
