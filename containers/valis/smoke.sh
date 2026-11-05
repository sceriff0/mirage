#!/usr/bin/env bash
# Smoke test for bolt3x/mirage-valis (the pipeline's VALIS 1.2.0 image, built FROM
# upstream's cdgatenbee/valis-wsi:1.2.0 with only the four permission/asset fixes).
#
# ONE DEFINITION, RUN IN TWO PLACES: the Dockerfile RUNs it as a build step, and
# .github/workflows/containers.yml runs it again with `docker run` against the built image.
#
# AS THE CLUSTER RUNS IT: every VALIS check below runs as an unprivileged user (uid 65534)
# when the script itself is root, from /, with HOME pointing at an empty directory, and
# with every proxy variable pointed at a closed local port, so a lookup that misses a baked
# file and goes to the network fails the BUILD, not a cluster task.

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

# 2. UPSTREAM'S STACK IS UPSTREAM'S, line for line: this image adds no Python package.
"$PY" -m pip freeze --all | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' | LC_ALL=C sort > "$SCRATCH/now.txt"
if ! LC_ALL=C diff /opt/mirage/upstream-freeze.txt "$SCRATCH/now.txt" >&2; then
  echo "the Python stack differs from upstream's freeze (diff above)" >&2
  exit 1
fi
echo "upstream's $(wc -l < /opt/mirage/upstream-freeze.txt) pinned distributions are untouched"

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

# 3. What REGISTER's own scripts import beside VALIS (bin/register.py and its bin/utils
# helpers: numpy, tifffile, scyjava's config for the read-only-$HOME JVM cache guard).
as_user "$PY" -c "
import numpy, tifffile
from scyjava import config as sjconf
assert hasattr(sjconf, 'set_cache_dir') and hasattr(sjconf, 'set_m2_repo')
print('REGISTER helpers OK: numpy', numpy.__version__, 'tifffile', tifffile.__version__)
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
