#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-merge.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/merge_channels_pyramid.py and
# bin/extract_mask_series.py, through bin/utils/tiled_io.py (numpy/tifffile/zarr).
# imagecodecs is named because the whole apt block above it exists to build it, and
# a tifffile write of an LZW-compressed pyramid fails without it.
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "
import numpy, tifffile, zarr, imagecodecs
print('merge image OK:',
      'numpy', numpy.__version__,
      '| tifffile', tifffile.__version__,
      '| zarr', zarr.__version__,
      '| imagecodecs', imagecodecs.__version__)"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
