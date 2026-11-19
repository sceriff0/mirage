#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-quantify.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/quantify.py, extract_cell_properties.py,
# export_geojson.py, mask_to_geojson.py and generate_postprocessing_qc.py. The
# matplotlib check forces the Agg backend FIRST: this image has no display, and
# importing pyplot without it is the failure generate_postprocessing_qc.py would hit.
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "
import numpy, pandas, scipy, tifffile, zarr, imagecodecs
import skimage.measure, skimage.segmentation, skimage.transform
from scipy import ndimage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot
print('quantify image OK:',
      'numpy', numpy.__version__,
      '| pandas', pandas.__version__,
      '| scipy', scipy.__version__,
      '| skimage', skimage.__version__,
      '| tifffile', tifffile.__version__,
      '| matplotlib', matplotlib.__version__,
      '| ndimage', ndimage.__name__)"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
