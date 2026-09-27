#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-drape.
#
# ONE DEFINITION, RUN IN TWO PLACES: this image's own Dockerfile RUNs it as a build
# step, and .github/workflows/containers.yml runs it again with `docker run` against
# the built image on a pull request that touched this build context. A stack that
# resolves in pip and then explodes on first import therefore fails the BUILD, not a
# cluster task six hours in -- scikit-image 0.25 against numpy 1.26 is a pairing pip
# will happily accept and then break on.
#
# WHAT IT ASSERTS, AND WHY THAT SET. Exactly the third-party modules the pipeline's
# own bin/ scripts import when they run in THIS image -- bin/tiled_coarse.py,
# tiled_reg_tile.py, tiled_solve.py and tiled_stitch.py, i.e. the `drape` package they
# shim over (numpy, scipy, scikit-image, tifffile, zarr, and imagecodecs for the zstd/LZW
# codecs tifffile writes through). Every name below is imported on a live run, so a name
# that stops importing is a broken image by definition. Do not add a module this image's
# processes never import; do not drop one they do.
#
# AND ONE NEGATIVE: torch must NOT import. This image exists because the retired
# DISK + LightGlue COARSE front-end no longer needs torch/kornia; a wheel that comes back
# (a transitive pull, a stray requirements line) would silently re-grow the image by ~1 GB.
set -euo pipefail

# `python` is NOT present in every base used here -- containers/convert installs
# python3 with no `python` alternative -- so resolve the interpreter rather than
# assuming a name.
PY="$(command -v python || command -v python3)"

"$PY" -c "import numpy, scipy, skimage, tifffile, zarr, imagecodecs; \
print('drape image OK:', numpy.__version__, scipy.__version__, skimage.__version__, tifffile.__version__, zarr.__version__, imagecodecs.__version__)"

# The method itself: the `drape` package (packages/drape, pip-installed by the Dockerfile),
# which bin/tiled_*.py shim over. A stage module that stops importing is a broken image
# just as a missing wheel is; importing all four proves the package AND its stage graph.
"$PY" -c "import drape, drape.stages.coarse, drape.stages.reg_tile, drape.stages.solve, drape.stages.stitch; \
print('drape package OK:', drape.__version__)"

# The negative: no learned stack in this image. Written as __import__() so the smoke
# scanner (tests/test_container_smoke_tests.py), which requires every `import X` here to be
# installed, does not read it as a positive import.
for mod in torch kornia; do
  if "$PY" -c "__import__('${mod}')" 2>/dev/null; then
    echo "${mod} is importable in the slim DRAPE image -- it must not be" >&2
    exit 1
  fi
done
echo "no torch/kornia: OK"

# procps supplies `ps`; Nextflow's task-metrics wrapper hard-exits without it.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
