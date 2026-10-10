#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-valis120 (the registration benchmark's VALIS 1.2.0
# image).
#
# ONE DEFINITION, RUN IN TWO PLACES: this image's own Dockerfile RUNs it as a build step,
# and .github/workflows/containers.yml runs it again with `docker run` against the built
# image. Nothing in requirements/valis120.txt was installed on the machine that wrote it,
# so THIS is where the stack is first proven to import.
#
# NO NETWORK, AND THAT IS THE POINT OF THIS SCRIPT. `import valis` builds VALIS's default
# matcher, which loads the DISK and LightGlue weights through torch.hub, and starting its
# JVM asks Maven for Bio-Formats unless the jar is beside the package. Both were put into
# the image by fetch_assets.py. Every proxy variable points at a closed local port for
# the whole script, so if either lookup misses the baked file and goes to the network, it
# is refused at once and the BUILD fails -- not a cluster task on a node without network.
# (The Dockerfile runs this inside `docker build`, where the network cannot be switched
# off from within the script; a refused proxy is the equivalent that works in both places.)
set -euo pipefail

export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9
export no_proxy="" NO_PROXY=""

PY="$(command -v python || command -v python3)"

# What benchmarks/regbench imports in every method's image, and pytest for the first
# cluster command (test_external_methods.py runs inside the image).
"$PY" -c "import numpy, scipy, skimage, tifffile, zarr, numcodecs, imagecodecs, pandas, psutil, pytest; \
print('valis120 image OK:', numpy.__version__, scipy.__version__, skimage.__version__, tifffile.__version__, zarr.__version__)"

# The learned stack VALIS 1.2.0 requires, as the CPU build, and libvips from its wheel.
"$PY" -c "
import torch, torchvision, kornia, pyvips
assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert torch.version.cuda is None, 'the CUDA wheel was installed: ' + str(torch.version.cuda)
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'kornia', kornia.__version__, 'libvips', pyvips.version(0), pyvips.version(1))
"

# The baked files are where the two lookups look, readable by any user.
"$PY" -c "
import os, pathlib
home = os.environ.get('TORCH_HOME')
assert home == '/opt/torch', home
for name in ('depth-save.pth', 'disk_lightglue_v0-1_arxiv-pth'):
    f = pathlib.Path(home, 'hub', 'checkpoints', name)
    assert f.is_file() and f.stat().st_size > 1000000, f
    assert f.stat().st_mode & 0o004, f
print('weights OK:', sorted(p.name for p in pathlib.Path(home, 'hub', 'checkpoints').iterdir()))
"

# THE IMPORT THAT DOWNLOADS, with the network closed. It must find both weight files.
"$PY" -c "
import valis
from valis import registration, slide_io
assert valis.__version__ == '1.2.0', valis.__version__
matcher = registration.DEFAULT_MATCHER
print('valis', valis.__version__, 'default matcher', type(matcher).__name__, 'on', type(getattr(matcher, 'feature_detector', None)).__name__)
"

# THE JVM, with the network closed: it must start from the bundled jar (a Maven lookup
# would be refused), and the jar must be the pinned Bio-Formats. Starting it also rewrites
# <package>/data/bf_formats.txt, so that file and its directory are checked afterwards
# for the permissions another user needs to do the same.
"$PY" -c "
import os, pathlib
from valis import slide_io
package = pathlib.Path(slide_io.__file__).parent
assert (package / 'bioformats_package.jar').is_file(), 'the jar is not beside the installed package'
slide_io.init_jvm()
version = str(slide_io.get_bioformats_version())
slide_io.kill_jvm()
assert version == '7.0.0', version
data = package / 'data'
assert data.stat().st_mode & 0o003 == 0o003, oct(data.stat().st_mode)
assert (data / 'bf_formats.txt').stat().st_mode & 0o002, 'bf_formats.txt is not writable by others'
assert (data / 'bf_formats.txt').stat().st_size > 0
print('Bio-Formats', version, 'from the bundled jar; data directory writable by any user')
"

# STARE: the selfcheck and the adapter read images through stare.slide_io.
"$PY" -c "
import stare
from stare.slide_io import open_lazy
from stare.ome import read_ome_pixel_size
print('stare', stare.__version__)
"

# procps supplies `ps`; every image in containers/ carries it.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
