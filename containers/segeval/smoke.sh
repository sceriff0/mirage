#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-segeval.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/seg_quality_eval.py and the vendored
# CellSegmentationEvaluator subset under bin/utils/cse/.
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "\
import numpy, scipy, skimage, sklearn, pandas, tifffile, xmltodict; \
from sklearn.cluster import KMeans; \
from skimage.segmentation import find_boundaries; \
print('segeval image OK:', \
      'numpy', numpy.__version__, \
      '| scipy', scipy.__version__, \
      '| skimage', skimage.__version__, \
      '| sklearn', sklearn.__version__, \
      '| tifffile', tifffile.__version__)"

"$PY" -c "\
import matplotlib; matplotlib.use('Agg'); \
import matplotlib.pyplot; \
print('matplotlib OK -- functions.py foreground_separation() can import pyplot')"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
