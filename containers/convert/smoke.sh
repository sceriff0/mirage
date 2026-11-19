#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-convert.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the modules bin/convert_image.py imports (numpy, tifffile, bioio, h5py) plus the
# Bio-Formats reader: bioio finds plugins through an entry point, so a broken
# bioio-bioformats is not an ImportError, just a format that silently cannot be read.
#
# The positive leg reads a file with HOME=/nonexistent through the production redirect
# (bin/utils/jvm_cache.py, copied in by the Dockerfile). The negative leg repeats the
# read as an unprivileged uid without the redirect and must fail; root alone would pass
# because it can write anywhere.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "
import numpy, tifffile, h5py
from bioio import BioImage
import bioio_bioformats
print('convert image OK:',
      'numpy', numpy.__version__,
      '| tifffile', tifffile.__version__,
      '| h5py', h5py.__version__,
      '| bioio reader', BioImage.__name__,
      '| bioio-bioformats', bioio_bioformats.__name__)"

# The baked Maven artifacts. jgo creates the directory before filling it, so check it is
# not empty.
test -d /root/.m2/repository/ome/formats-gpl
test -n "$(ls -A /root/.m2/repository/ome/formats-gpl)"
test -d /root/.jgo
echo "convert image OK: Bio-Formats jars are baked under /root/.m2 and /root/.jgo"

# Positive leg: read through Bio-Formats with no usable $HOME, via the production redirect.
PYTHONPATH=/usr/local/lib/mirage_jvm HOME=/nonexistent "$PY" -c "
from jvm_cache import point_jvm_cache_off_readonly_home
point_jvm_cache_off_readonly_home()
import numpy, tifffile
tifffile.imwrite('/tmp/smoke-probe.tif', numpy.zeros((8, 8), 'uint16'))
from bioio_bioformats import Reader
print('Bio-Formats reads with no HOME (production redirect):',
      Reader('/tmp/smoke-probe.tif').dims)"

# Remove the probe: left behind as a root-owned file, it makes a later run of this script
# under another uid fail on the write before reaching the redirect.
rm -f /tmp/smoke-probe.tif

# Negative leg: same read without the redirect, as an unprivileged uid; must fail.
set +e
HOME=/nonexistent "$PY" -c "
import os
# Drop privilege only when we STARTED as root (the Dockerfile's build-time RUN). When
# this script is instead invoked at RUNTIME already under a non-root uid -- Task 13's
# \`docker run -u 1000:1000\` check, and nextflow.config's own \`-u \$(id -u):\$(id -g)\` --
# os.setgid/os.setuid(65534) would try to change to a DIFFERENT non-root uid, which
# needs CAP_SETUID and raises a plain PermissionError (EPERM) on its own, before the
# JVM-cache path below is ever reached. That EPERM still makes this leg exit non-zero,
# so the smoke test stays green -- but for the wrong reason: it stops proving the
# read-only-\$HOME guard is load-bearing and starts merely proving setuid(other-uid)
# is refused. Any already-non-root uid IS the unprivileged condition this leg wants
# (mkdir under /, root-owned mode 755, refuses every non-root uid alike), so there is
# nothing left to drop. Measured red (EPERM at this line) under \`-u 1000:1000\` before
# this guard, green (the intended EACCES/read-only chain) after, 2026-09-03.
if os.getuid() == 0:
    os.setgid(65534)
    os.setuid(65534)
import numpy, tifffile
tifffile.imwrite('/tmp/smoke-noguard-probe.tif', numpy.zeros((8, 8), 'uint16'))
from bioio_bioformats import Reader
Reader('/tmp/smoke-noguard-probe.tif')
"
noguard_status=$?
set -e
if [ "$noguard_status" -eq 0 ]; then
    echo "convert image FAIL: Bio-Formats read SUCCEEDED under HOME=/nonexistent with no jvm-cache redirect and no root DAC override -- the redirect is not load-bearing, or this leg is not exercising the failure it claims to" >&2
    exit 1
fi
echo "convert image OK: Bio-Formats read correctly FAILS (exit $noguard_status) with no jvm-cache redirect -- the positive leg's redirect above is load-bearing, not decorative"

# Written by uid 65534, so remove it for the same reason (the outer shell is still root).
rm -f /tmp/smoke-noguard-probe.tif

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
