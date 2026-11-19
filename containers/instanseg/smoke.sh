#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-instanseg.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/segment_instantseg.py, whose InstanSeg import
# is lazy (inside the function) and so is never exercised by the pip resolve alone.
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "
import numpy, tifffile
from instanseg import InstanSeg
print('instanseg image OK:',
      'numpy', numpy.__version__,
      '| tifffile', tifffile.__version__,
      '| InstanSeg', InstanSeg.__name__)"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
