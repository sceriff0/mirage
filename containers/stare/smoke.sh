#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-stare.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/tiled_coarse.py,
# tiled_reg_tile.py, tiled_solve.py and tiled_stitch.py, i.e. the `stare` package they
# shim over (numpy, scipy, scikit-image, tifffile, zarr, and imagecodecs for the zstd/LZW
# codecs tifffile writes through). Do not add a module these scripts never import; do not
# drop one they do.
#
# Also asserts torch and kornia do not import, so a transitive pull cannot regrow the image.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "import numpy, scipy, skimage, tifffile, zarr, imagecodecs; \
print('stare image OK:', numpy.__version__, scipy.__version__, skimage.__version__, tifffile.__version__, zarr.__version__, imagecodecs.__version__)"

# The `stare` package (pinned in requirements/stare.txt) and its four stage modules.
"$PY" -c "import stare, stare.stages.coarse, stare.stages.reg_tile, stare.stages.solve, stare.stages.stitch; \
print('stare package OK:', stare.__version__)"

# No learned stack. __import__() so tests/test_container_smoke_tests.py does not read it
# as a positive import.
for mod in torch kornia; do
  if "$PY" -c "__import__('${mod}')" 2>/dev/null; then
    echo "${mod} is importable in the slim STARE image -- it must not be" >&2
    exit 1
  fi
done
echo "no torch/kornia: OK"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
