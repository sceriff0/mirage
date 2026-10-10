#!/usr/bin/env python3
"""Put upstream's own baked files where VALIS 1.2.0 looks for them, for any user.

Run once by containers/valis120/Dockerfile, as a build step. It needs NO network: every
file it places is already in upstream's image (cdgatenbee/valis-wsi:1.2.0), in a place a
user who is not root, on a compute node, cannot use. After it, smoke.sh imports VALIS and
starts its JVM as an unprivileged user with every proxy variable pointed at a closed port.

WHAT UPSTREAM'S IMAGE HOLDS, AND HOW THAT WAS LEARNED. Not from a running container
(no Docker daemon where this was written) but from Docker Hub's registry API on
2026-10-11: the image index, the linux/amd64 manifest, its config (environment, build
history) and the file lists of its layers, streamed and listed without unpacking
[PARTIAL: config and history in full; names, modes and link targets of layers 1, 3, 4,
5, 8, 21 to 26 of 26; the dist-info names of layer 22; no file contents except
pyproject.toml, docker_download_weights.py, pyvenv.cfg and the sha256 of the files named
below].

1. Python is a virtual environment at ``/root/.local`` (``python3 -m venv ~/.local`` in
   the build history; ``pyvenv.cfg``: 3.12.3), and ``/root`` is mode 0700 in every layer
   that lists it. ``uv pip install .`` put VALIS and its stack into that environment
   (``valis_wsi-1.2.0.dist-info/direct_url.json``: ``file:///usr/local/src``). As
   published, the image therefore has NO Python for a user who is not root. The
   Dockerfile opens ``/root`` for reading; this script checks it.

2. The Bio-Formats jar is at ``/usr/local/src/valis/bioformats_package.jar`` (the build
   history's ``wget .../bio-formats/7.0.0/artifacts/bioformats_package.jar -P valis``, run
   in ``/usr/local/src``): beside the SOURCE tree, not beside the installed package.
   ``valis.slide_io.init_jvm`` (valis-wsi 1.2.0 from PyPI, ``slide_io.py:231-262``
   [PARTIAL: lines 87-300 read]) starts the JVM from
   ``<directory of slide_io.py>/bioformats_package.jar`` when that file exists and
   otherwise asks Maven. From ``/usr/local/src`` (the image's working directory)
   ``import valis`` finds the source tree, so in Docker the jar is found; from any other
   directory it finds the installed copy, which has no jar. The benchmark runs from the
   repository, so the jar is linked beside the installed package. Its sha256 is the one
   OME publishes (``.../7.0.0/artifacts/SHASUMS`` [FULL], read by the Task 11 agent on
   2026-10-10) and the one the layer's file has (hashed from the registry stream).

3. The weights are in root's own cache, ``/root/.cache/torch/hub/checkpoints``
   (upstream's ``docker_download_weights.py`` builds ``DiskFD``, ``DeDoDeFD`` and both
   LightGlue matchers at build time): six files. ``import valis`` builds the default
   matcher (``registration.py:72``; DISK through ``kornia.feature.DISK.from_pretrained``,
   LightGlue through ``kornia.feature.LightGlueMatcher('disk')``), and both go through
   ``torch.hub.load_state_dict_from_url``, which looks in ``$TORCH_HOME/hub/checkpoints``
   or, with no ``TORCH_HOME``, in ``~/.cache/torch`` OF WHOEVER IS RUNNING: not root's.
   The image sets no ``TORCH_HOME``. The Dockerfile sets ``TORCH_HOME=/opt/torch`` and
   this script links ``/opt/torch/hub/checkpoints`` to upstream's directory, after
   checking the two files the default matcher loads against the sha256 of the files at
   their published URLs (computed by the Task 11 agent on 2026-10-10 from
   ``raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth`` and
   ``github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth``; the
   layer's files hash to the same values).

4. ``<package>/data/bf_formats.txt`` is REWRITTEN by ``init_jvm`` on every JVM start
   (``slide_io.py:95, 274-279``; new in 1.2.0). It is made writable by any user, in the
   installed package and in the source tree. THAT IS NOT ENOUGH UNDER APPTAINER, where
   the image is mounted read-only: the run must add ``--writable-tmpfs``
   (containers/README.md says so in this image's row).

Links, not copies: the files stay upstream's own, and the image does not grow.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
from pathlib import Path

SOURCE_TREE = Path("/usr/local/src/valis")
UPSTREAM_CHECKPOINTS = Path("/root/.cache/torch/hub/checkpoints")
JAR = (
    "bioformats_package.jar",
    "a38e1c9efb91f134591fe2c86d731b84f54765a99883f4120ffedea913a121a9",  # 7.0.0
)
WEIGHTS = (
    (
        "depth-save.pth",
        "9c2ee4ded238892dfa51569941372601e35e4a74aa6f84ea80053d2ab1c07abe",
    ),
    (
        "disk_lightglue_v0-1_arxiv-pth",
        "b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335",
    ),
)
READ_ALL = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
WRITE_ALL = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
ENTER_ALL = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH


def package_dir():
    """Where the INSTALLED ``valis`` is, found without importing it (the import builds
    the matcher) and without the working directory on the path (in ``/usr/local/src``
    the source tree would answer)."""
    here = {"", ".", os.getcwd()}
    sys.path[:] = [p for p in sys.path if p not in here]
    spec = importlib.util.find_spec("valis")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("valis is not installed in this interpreter")
    package = Path(list(spec.submodule_search_locations)[0])
    if "site-packages" not in package.parts:
        raise SystemExit(f"{package} is not an installed package (no site-packages)")
    return package


def checkpoints_dir():
    home = os.environ.get("TORCH_HOME")
    if not home:
        raise SystemExit(
            "TORCH_HOME is not set: torch.hub would look in the home of whoever runs "
            "the image, and the weights are in root's"
        )
    return Path(home) / "hub" / "checkpoints"


def check(path, sha256):
    """Refuse a baked file that is not the one this image was written for."""
    if not path.is_file():
        raise SystemExit(f"{path} is not in the base image: it is not upstream's 1.2.0")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise SystemExit(
            f"{path}\n  sha256 {digest.hexdigest()}\n  wanted {sha256}\n"
            "The file in the base image is not the one this image was written for."
        )
    print(f"{path}  {path.stat().st_size} bytes  sha256 {sha256}")


def readable_by_anyone(path):
    """Every directory from ``/`` down to ``path`` lets any user in, and ``path`` itself
    can be read: what a user who is not root needs to reach it."""
    for folder in [*reversed(path.parents)]:
        mode = folder.stat().st_mode
        if mode & (stat.S_IROTH | stat.S_IXOTH) != (stat.S_IROTH | stat.S_IXOTH):
            raise SystemExit(
                f"{folder} is mode {oct(mode & 0o777)}: a user who is not root cannot "
                f"reach {path}. The Dockerfile's `chmod a+rx /root` must run first."
            )
    if not path.stat().st_mode & stat.S_IROTH:
        raise SystemExit(f"{path} is not readable by other users")


def open_up(path, extra):
    path.chmod(path.stat().st_mode | READ_ALL | extra)


def link(target, name):
    name.parent.mkdir(parents=True, exist_ok=True)
    if name.is_symlink() or name.exists():
        raise SystemExit(f"{name} already exists: the base image is not as it was read")
    name.symlink_to(target)
    print(f"{name} -> {target}")


def main():
    package = package_dir()
    checkpoints = checkpoints_dir()  # refuses before anything is touched
    readable_by_anyone(Path(sys.executable))  # NOT resolved: the venv is under /root
    readable_by_anyone(package / "__init__.py")

    jar, sha256 = JAR
    check(SOURCE_TREE / jar, sha256)
    readable_by_anyone(SOURCE_TREE / jar)
    link(SOURCE_TREE / jar, package / jar)

    for name, sha256 in WEIGHTS:
        check(UPSTREAM_CHECKPOINTS / name, sha256)
        readable_by_anyone(UPSTREAM_CHECKPOINTS / name)
    link(UPSTREAM_CHECKPOINTS, checkpoints)
    for folder in (checkpoints.parent, checkpoints.parent.parent):  # hub, TORCH_HOME
        open_up(folder, ENTER_ALL)

    for data in (package / "data", SOURCE_TREE / "data"):
        formats = data / "bf_formats.txt"
        if not formats.is_file():
            raise SystemExit(
                f"{formats} is missing: valis.slide_io reads it at import "
                "(slide_io.py:104-111)"
            )
        open_up(data, ENTER_ALL | WRITE_ALL)
        open_up(formats, WRITE_ALL)
        print(f"{data} and {formats.name}: writable by any user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
