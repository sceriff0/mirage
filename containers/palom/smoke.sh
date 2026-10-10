#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-palom (the registration benchmark's PALOM image).
#
# ONE DEFINITION, RUN IN TWO PLACES: this image's own Dockerfile RUNs it as a build step,
# and .github/workflows/containers.yml runs it again with `docker run` against the built
# image. Nothing in requirements/palom.txt was installed on the machine that wrote it, so
# THIS is where the stack is first proven to import.
#
# WHAT IT ASSERTS. The pinned PALOM release and the two versions PALOM caps; what
# benchmarks/regbench imports in this image (numpy, scipy, tifffile, zarr, dask, psutil,
# and pytest for the first cluster command, test_external_methods.py); and the two STARE
# modules the benchmark reads images through, which were installed WITHOUT STARE's
# dependencies and therefore have to be shown to import on PALOM's stack.
set -euo pipefail

PY="$(command -v python || command -v python3)"

"$PY" -c "import numpy, scipy, skimage, tifffile, zarr, numcodecs, imagecodecs, dask, psutil, pytest; \
print('palom image OK:', numpy.__version__, scipy.__version__, skimage.__version__, tifffile.__version__, zarr.__version__, dask.__version__)"

# PALOM itself, at the pinned release, on the versions its metadata caps. __import__ for
# cv2: its distribution here is opencv-python-headless, which the smoke scanner
# (tests/test_container_smoke_tests.py) would look up as opencv-python.
"$PY" -c "
import palom, skimage, tifffile
import palom.align, palom.reader, palom.pyramid, palom.register
assert palom.__version__ == '2026.4.1', palom.__version__
assert skimage.__version__ == '0.19.3', skimage.__version__
assert tifffile.__version__ == '2024.7.2', tifffile.__version__
print('palom', palom.__version__, 'cv2', __import__('cv2').__version__)
"

# dask sizes its thread pool from the CPU affinity mask only when psutil is importable
# (benchmarks/regbench/methods/palom.py records which it saw); prove the count resolves.
"$PY" -c "
import dask.system, psutil
assert dask.system.CPU_COUNT >= 1
print('dask CPU_COUNT', dask.system.CPU_COUNT, 'affinity', len(psutil.Process().cpu_affinity()))
"

# STARE's reader modules, installed with --no-deps: they must import here although
# STARE's own scikit-image requirement is not met in this image.
"$PY" -c "
import stare
import stare.slide_io, stare.ome
from stare.slide_io import open_lazy
from stare.ome import read_ome_pixel_size
print('stare readers OK:', stare.__version__)
"

# procps supplies `ps`; every image in containers/ carries it.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
