#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-deeperhistreg (the registration benchmark's
# DeeperHistReg image).
#
# ONE DEFINITION, RUN IN TWO PLACES: this image's own Dockerfile RUNs it as a build step,
# and .github/workflows/containers.yml runs it again with `docker run` against the built
# image. Most of requirements/deeperhistreg.txt was never installed on the machine that
# wrote it, so THIS is where the stack is first proven to import.
#
# WHAT IT ASSERTS. The pinned package and the libraries it loads (torch, torchvision,
# SimpleITK, pyvips); that the torch imported is still the base image's CUDA build and
# not a wheel pip put over it; what benchmarks/regbench imports in this image (numpy,
# scipy, scikit-image, tifffile, zarr, psutil, STARE's reader, and pytest for the first
# cluster command, test_external_methods.py).
#
# NO NETWORK. The benchmark runs on compute nodes that may have none. Every proxy
# variable points at a closed local port for the whole script, so an import that tries to
# download anything fails here, at build time, instead of on the cluster.
set -euo pipefail

export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9
export no_proxy="" NO_PROXY=""

PY="$(command -v python || command -v python3)"

"$PY" -c "import numpy, scipy, skimage, tifffile, zarr, numcodecs, imagecodecs, pandas, psutil, pytest; \
print('deeperhistreg image OK:', numpy.__version__, scipy.__version__, skimage.__version__, tifffile.__version__, zarr.__version__)"

# The framework: the base image's CUDA build, at or above the benchmark's floor. A wheel
# from PyPI installed over it would still import, so the build label is asserted.
"$PY" -c "
import torch, torchvision
major, minor = (int(x) for x in torch.__version__.split('.')[:2])
assert (major, minor) >= (2, 3), torch.__version__
assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith('12.4'), torch.version.cuda
assert torch.from_numpy(__import__('numpy').zeros((2, 2), dtype='uint16')).dtype == torch.uint16
print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'torchvision', torchvision.__version__)
"

# The package itself and the libraries it loads. The version is read from the installed
# metadata, as the adapter does when it enforces 1.0.1 (written with __import__ because
# the smoke scanner, tests/test_container_smoke_tests.py, has no entry for that standard
# library module and would look it up as a distribution).
"$PY" -c "
import deeperhistreg, SimpleITK, pyvips
found = __import__('importlib.metadata').metadata.version('deeperhistreg')
assert found == '1.0.1', found
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid, default_initial_nonrigid_high_resolution
print('deeperhistreg 1.0.1, SimpleITK', SimpleITK.__version__, 'libvips', pyvips.version(0), pyvips.version(1), 'cv2', __import__('cv2').__version__)
"

# STARE: the adapter reads images through stare.slide_io.
"$PY" -c "
import stare
from stare.slide_io import open_lazy
from stare.ome import read_ome_pixel_size
print('stare', stare.__version__)
"

# procps supplies `ps`; every image in containers/ carries it.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
