#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-regqc.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/generate_registration_qc.py, through
# bin/utils/qc.py, which imports cv2 at MODULE scope. opencv comes from
# requirements/regqc.txt as the NON-headless build because this image carries the
# display libs, and a headless/non-headless mixup breaks only at import time.
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "
import numpy, tifffile, cv2
import skimage.transform
print('regqc image OK:',
      'numpy', numpy.__version__,
      '| tifffile', tifffile.__version__,
      '| cv2', cv2.__version__,
      '| skimage', skimage.__version__)"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
