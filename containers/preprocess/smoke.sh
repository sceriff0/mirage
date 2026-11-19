#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-preprocess.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/apply_basic_profiles.py, tile_for_basic.py,
# split_multichannel.py and generate_preprocess_qc.py, through bin/utils/tiled_io.py.
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "
import numpy, scipy, tifffile, zarr, imagecodecs
import skimage.io, skimage.transform
print('preprocess image OK:',
      'numpy', numpy.__version__,
      '| scipy', scipy.__version__,
      '| skimage', skimage.__version__,
      '| tifffile', tifffile.__version__,
      '| zarr', zarr.__version__,
      '| imagecodecs', imagecodecs.__version__)"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
