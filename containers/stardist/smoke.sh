#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-stardist.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/segment.py, which imports csbdeep and
# stardist.models at MODULE scope and reads its input through dask.array.
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "
import numpy, tifffile, dask.array, skimage
from csbdeep.utils import normalize
from stardist.models import StarDist2D
print('stardist image OK:',
      'numpy', numpy.__version__,
      '| tifffile', tifffile.__version__,
      '| skimage', skimage.__version__,
      '| csbdeep normalize', normalize.__name__,
      '| StarDist2D', StarDist2D.__name__)"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
