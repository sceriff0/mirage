#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-stare-regbench (the registration benchmark's own
# STARE image: the `stare` run, `tissue` and `score`).
#
# ONE DEFINITION, RUN IN TWO PLACES: this image's own Dockerfile RUNs it as a build step,
# and .github/workflows/containers.yml runs it again with `docker run` against the built
# image. Nothing in requirements/stare-regbench.txt was installed on the machine that
# wrote it, so THIS is where the stack is first proven to import.
#
# WHAT IT ASSERTS. STARE and its stage graph, as containers/stare/smoke.sh does (the
# method here must be the pipeline image's). Then what the benchmark adds: psutil with a
# readable resident-set size (the reason this image exists), pytest, and OpenCV at the
# pinned version with the calls the tissue-level block makes. And the same negative as
# the pipeline image: no torch, no kornia.
set -euo pipefail

PY="$(command -v python || command -v python3)"

"$PY" -c "import numpy, scipy, skimage, tifffile, zarr, numcodecs, imagecodecs, psutil, pytest; \
print('stare-regbench image OK:', numpy.__version__, scipy.__version__, skimage.__version__, tifffile.__version__, zarr.__version__, imagecodecs.__version__)"

# The method itself: the pinned release of github.com/sceriff0/stare, all four stages,
# and the reader every part of the benchmark opens images through.
"$PY" -c "
import stare, stare.stages.coarse, stare.stages.reg_tile, stare.stages.solve, stare.stages.stitch
from stare.slide_io import open_lazy
from stare.ome import read_ome_pixel_size
print('stare package OK:', stare.__version__)
"

# STARE 1.3.0 or later: the stage commands take --prep (the benchmark's `sprep` run option
# passes it to `stare coarse` and `stare reg-tile`).
"$PY" -m stare.cli coarse --help | grep -q -- --prep \
  || { echo "stare coarse has no --prep: the image carries a STARE older than 1.3.0" >&2; exit 1; }
"$PY" -m stare.cli reg-tile --help | grep -q -- --prep

# psutil, doing what resources.Monitor asks of it: the resident set of this process and
# of its children, and the CPU affinity mask.
"$PY" -c "
import os, psutil
me = psutil.Process(os.getpid())
assert me.memory_info().rss > 0
me.children(recursive=True)
assert len(me.cpu_affinity()) >= 1
print('psutil', psutil.__version__, 'rss MB', me.memory_info().rss >> 20)
"

# OpenCV, at the one version a whole result is computed with, and the calls
# benchmarks/regbench/tissue.py makes. __import__ because the distribution here is
# opencv-python-headless, which the smoke scanner (tests/test_container_smoke_tests.py)
# would look up as opencv-python.
"$PY" -c "
import numpy
cv2 = __import__('cv2')
assert cv2.__version__ == '4.10.0', cv2.__version__
for name in ('ORB_create', 'BFMatcher', 'estimateAffinePartial2D', 'findContours', 'threshold', 'morphologyEx', 'getStructuringElement', 'connectedComponentsWithStats', 'dilate', 'remap', 'warpAffine'):
    assert hasattr(cv2, name), name
orb = cv2.ORB_create(nfeatures=500)
image = (numpy.random.default_rng(0).random((256, 256)) * 255).astype('uint8')
points, _ = orb.detectAndCompute(image, None)
print('cv2', cv2.__version__, 'ORB keypoints on noise:', len(points))
"

# The negative: no learned stack in this image either. __import__() so the smoke scanner
# does not read it as a positive import.
for mod in torch kornia; do
  if "$PY" -c "__import__('${mod}')" 2>/dev/null; then
    echo "${mod} is importable in the benchmark's STARE image -- it must not be" >&2
    exit 1
  fi
done
echo "no torch/kornia: OK"

# procps supplies `ps`; every image in containers/ carries it.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
