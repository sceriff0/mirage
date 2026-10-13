#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-valis100 (the registration benchmark's VALIS 1.0.0 image).
#
# ONE DEFINITION, RUN IN TWO PLACES: the Dockerfile RUNs it as a build step, and
# .github/workflows/containers.yml runs it again with `docker run` against the built
# image. Nothing here was installed on the machine that wrote it, so THIS is where the
# added codecs are first proven to work on upstream's stack.
#
# WHAT IT ASSERTS. That VALIS 1.0.0 still imports at its release; that what the benchmark's
# preflight asks of every image imports (numpy, tifffile, zarr, numcodecs, imagecodecs);
# that a compressed, tiled TIFF round-trips through tifffile's zarr view, which is the
# read stare.slide_io.open_lazy makes; and that no line of upstream's freeze changed.
set -euo pipefail

PY="$(command -v python || command -v python3)"

"$PY" -c "import numpy, tifffile, zarr, numcodecs, imagecodecs, pytest; \
print('valis100 image OK:', numpy.__version__, tifffile.__version__, zarr.__version__, numcodecs.__version__, imagecodecs.__version__)"

"$PY" -c "
import valis
from valis import registration
v = valis.__version__
assert v == '1.0.0', v
print('valis', v)
"

# The read the harness makes: an LZW, tiled, multi-channel TIFF through STARE's reader
# (tifffile's zarr view). STARE was installed without its dependencies, so this is also
# the proof that stare.slide_io imports and runs on upstream's numpy and tifffile.
"$PY" -c "
import os, numpy, tifffile, stare
from stare.slide_io import open_lazy
a = (numpy.arange(3 * 512 * 512, dtype='uint16') % 4099).reshape(3, 512, 512)
p = '/tmp/mirage-smoke-%d.tif' % os.getpid()
tifffile.imwrite(p, a, tile=(256, 256), compression='lzw', photometric='minisblack')
arr, dtype, close = open_lazy(p)
assert numpy.array_equal(numpy.asarray(arr[1, 100:400, 50:300]), a[1, 100:400, 50:300])
close()
os.remove(p)
print('stare', stare.__version__, 'reads an lzw tiled TIFF through open_lazy')
"

# Nothing of upstream's stack moved: every line of the freeze taken before the install
# is still a line of the freeze now.
"$PY" -m pip freeze --all | grep -E '^[A-Za-z0-9._-]+==[^ ]+$' | LC_ALL=C sort > /tmp/freeze-now.txt
moved="$(LC_ALL=C comm -23 /opt/mirage/upstream-freeze.txt /tmp/freeze-now.txt)"
if [ -n "$moved" ]; then
    echo "upstream's stack changed; these lines of its freeze are gone:" >&2
    echo "$moved" >&2
    exit 1
fi
echo "upstream freeze intact: $(wc -l < /opt/mirage/upstream-freeze.txt) distributions"

ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
