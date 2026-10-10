#!/usr/bin/env bash
# Smoke test for bolt3x/mirage-valis120 (the registration benchmark's VALIS 1.2.0 image,
# built FROM upstream's cdgatenbee/valis-wsi:1.2.0).
#
# ONE DEFINITION, RUN IN TWO PLACES: this image's own Dockerfile RUNs it as a build step,
# and .github/workflows/containers.yml runs it again with `docker run` against the built
# image. Upstream's image was READ (Docker Hub's registry API), never run, on the machine
# that wrote this image, so THIS is where every statement of the Dockerfile's header is
# first tested.
#
# AS THE CLUSTER RUNS IT. Upstream's image works for root in /usr/local/src with root's
# home. A compute node runs it as another user, from another directory, with another
# $HOME, possibly without network. So every VALIS check below runs
#   - as an unprivileged user (uid 65534) when the script itself is root,
#   - from /, where `import valis` finds the INSTALLED package and not the source tree,
#   - with HOME pointing at an empty directory,
#   - with every proxy variable pointed at a closed local port, so a lookup that misses a
#     baked file and goes to the network is refused at once and the BUILD fails -- not a
#     cluster task. (The Dockerfile runs this inside `docker build`, where the network
#     cannot be switched off from within the script; a refused proxy is the equivalent
#     that works in both places.)
set -euo pipefail

export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9
export no_proxy="" NO_PROXY=""

PY="$(command -v python || command -v python3)"

SCRATCH="$(mktemp -d)"
chmod 0777 "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cd /

# Run a command as the user a cluster would: not root, no home to speak of. TORCH_HOME is
# NOT touched: it must be the image's own setting that finds the weights.
as_user() {
  local env=(env "HOME=$SCRATCH/home" "MPLCONFIGDIR=$SCRATCH/mpl" "XDG_CACHE_HOME=$SCRATCH/cache" "NUMBA_CACHE_DIR=$SCRATCH/numba")
  if [ "$(id -u)" = "0" ]; then
    setpriv --reuid=65534 --regid=65534 --clear-groups "${env[@]}" "$@"
  else
    "${env[@]}" "$@"
  fi
}

# 1. The interpreter is reachable at all (it lives under /root, see the Dockerfile).
as_user "$PY" -c "import sys; print('python', sys.version.split()[0], 'at', sys.executable, 'as uid', __import__('os').getuid())"

# 2. UPSTREAM'S STACK IS UPSTREAM'S. The freeze taken before the benchmark's additions
# must still hold line for line, except the one line the image changes on purpose.
"$PY" -m pip freeze --all | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' | LC_ALL=C sort > "$SCRATCH/now.txt"
moved="$(LC_ALL=C comm -23 /opt/mirage/upstream-freeze.txt "$SCRATCH/now.txt")"
if [ "$moved" != "tifffile==2025.6.11" ]; then
  echo "upstream's stack was changed beyond tifffile. Lines of the base image's freeze no longer installed:" >&2
  echo "$moved" >&2
  exit 1
fi
echo "upstream's $(wc -l < /opt/mirage/upstream-freeze.txt) pinned distributions are untouched but tifffile; added by the benchmark:"
LC_ALL=C comm -13 /opt/mirage/upstream-freeze.txt "$SCRATCH/now.txt" | sed 's/^/  /'

# The versions read from the registry, so that a rebuilt upstream tag is noticed here
# as well as by the digest.
as_user "$PY" -c "
import numpy, scipy, skimage, pandas, psutil, torch, torchvision, kornia, pyvips
got = dict(numpy=numpy.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
           pandas=pandas.__version__, psutil=psutil.__version__,
           torch=torch.__version__.split('+')[0], torchvision=torchvision.__version__.split('+')[0],
           kornia=kornia.__version__, pyvips=pyvips.__version__)
want = dict(numpy='1.26.4', scipy='1.16.0', skimage='0.25.2', pandas='2.3.0', psutil='7.0.0',
            torch='2.7.1', torchvision='0.22.1', kornia='0.8.1', pyvips='3.0.0')
assert got == want, {k: (got[k], want[k]) for k in want if got[k] != want[k]}
assert (pyvips.version(0), pyvips.version(1)) == (8, 16), (pyvips.version(0), pyvips.version(1))
print('upstream stack:', got, 'libvips', pyvips.version(0), pyvips.version(1), pyvips.version(2), 'torch cuda build', torch.version.cuda)
"

# 3. What the benchmark adds, and its reader on the downgraded tifffile.
as_user "$PY" -c "
import tifffile, zarr, numcodecs, imagecodecs, pytest
assert tifffile.__version__ == '2025.5.10', tifffile.__version__
assert zarr.__version__ == '2.18.3', zarr.__version__
import stare
from stare.slide_io import open_lazy
from stare.ome import read_ome_pixel_size
print('benchmark additions OK: tifffile', tifffile.__version__, 'zarr', zarr.__version__, 'numcodecs', numcodecs.__version__, 'imagecodecs', imagecodecs.__version__, 'pytest', pytest.__version__, 'stare', stare.__version__)
"

# STARE was installed with --no-deps: every requirement its own metadata states must be
# met by what is in the image (upstream's stack and the benchmark's additions).
as_user "$PY" -c "
from importlib import metadata
from packaging.requirements import Requirement
unmet = []
for line in metadata.requires('stare-registration') or []:
    req = Requirement(line)
    if req.marker is not None and not req.marker.evaluate({'extra': ''}):
        continue
    have = metadata.version(req.name)
    if not req.specifier.contains(have, prereleases=True):
        unmet.append((line, have))
assert not unmet, unmet
print('every requirement of stare-registration', metadata.version('stare-registration'), 'is met')
"

# The reader, doing what the benchmark asks of it: a tiled OME-TIFF written and then read
# back through tifffile's zarr view. This is the call newer tifffile refuses with zarr 2.
as_user "$PY" -c "
import os, numpy, tifffile
from stare.slide_io import open_lazy
path = os.path.join(os.environ['HOME'] + '-reader', 'probe.ome.tif')
os.makedirs(os.path.dirname(path))
image = numpy.arange(2 * 512 * 512, dtype='uint16').reshape(2, 512, 512)
tifffile.imwrite(path, image, tile=(256, 256), photometric='minisblack', metadata={'axes': 'CYX'})
array, dtype, close = open_lazy(path)
assert tuple(array.shape) == (2, 512, 512) and dtype == numpy.uint16, (array.shape, dtype)
assert (numpy.asarray(array[1, 200:300, 250:260]) == image[1, 200:300, 250:260]).all()
close()
print('stare.slide_io.open_lazy on tifffile', tifffile.__version__, 'with zarr 2: region read OK')
"

# 4. The baked files are where the two lookups look, for a user who is not root.
as_user "$PY" -c "
import os, pathlib
home = os.environ.get('TORCH_HOME')
assert home == '/opt/torch', home
for name in ('depth-save.pth', 'disk_lightglue_v0-1_arxiv-pth'):
    f = pathlib.Path(home, 'hub', 'checkpoints', name)
    assert f.is_file() and f.stat().st_size > 1000000, f
    with open(f, 'rb') as handle:
        assert handle.read(16)
print('weights readable as uid', os.getuid(), ':', sorted(p.name for p in pathlib.Path(home, 'hub', 'checkpoints').iterdir()))
"

# 5. THE IMPORT THAT DOWNLOADS, with the network closed, as that user. It must find both
# weight files, and it must be the installed package (from /, not /usr/local/src).
as_user "$PY" -c "
import valis
from valis import registration, slide_io
assert valis.__version__ == '1.2.0', valis.__version__
assert 'site-packages' in valis.__file__, valis.__file__
matcher = registration.DEFAULT_MATCHER
print('valis', valis.__version__, 'from', valis.__file__, 'default matcher', type(matcher).__name__, 'on', type(getattr(matcher, 'feature_detector', None)).__name__)
"

# 6. THE JVM, with the network closed, as that user: it must start from the jar beside
# the installed package (a Maven lookup would be refused), and the jar must be upstream's
# Bio-Formats 7.0.0. Starting it also rewrites <package>/data/bf_formats.txt, which is
# the write a user who is not root could not do in upstream's image.
as_user "$PY" -c "
import os, pathlib
from valis import slide_io
package = pathlib.Path(slide_io.__file__).parent
assert 'site-packages' in str(package), package
assert (package / 'bioformats_package.jar').is_file(), 'the jar is not beside the installed package'
before = (package / 'data' / 'bf_formats.txt').stat().st_mtime_ns
slide_io.init_jvm()
version = str(slide_io.get_bioformats_version())
slide_io.kill_jvm()
assert version == '7.0.0', version
formats = package / 'data' / 'bf_formats.txt'
assert formats.stat().st_size > 0
assert os.access(formats, os.W_OK) and os.access(package / 'data', os.W_OK | os.X_OK)
print('Bio-Formats', version, 'from the jar beside the installed package, as uid', os.getuid(), '; bf_formats.txt rewritten:', formats.stat().st_mtime_ns != before)
"

# procps supplies `ps`; every image in containers/ carries it.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
