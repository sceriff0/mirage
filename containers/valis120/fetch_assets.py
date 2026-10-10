#!/usr/bin/env python3
"""Put into the image the three files VALIS 1.2.0 would otherwise download at run time.

Run once by containers/valis120/Dockerfile, as a build step, with the network. After it,
nothing in this image needs the network: smoke.sh imports VALIS and starts its JVM with
every proxy variable pointed at a closed port.

WHY EACH FILE, WITH WHERE THAT WAS READ (read, not run: nothing of this stack was installed
on the machine that wrote this file).

1. The Bio-Formats jar. ``valis.slide_io.init_jvm`` (valis-wsi 1.2.0 from PyPI,
   ``slide_io.py:231-262`` [PARTIAL: lines 87-300 read]) starts the JVM from
   ``<package>/bioformats_package.jar`` when that file exists, "like in a Docker image",
   and otherwise asks scyjava to fetch ``ome:formats-gpl`` from Maven. The jar is version
   7.0.0, the ``BF_VERSION`` of upstream's own image ``cdgatenbee/valis-wsi:1.2.0`` (its
   build history, read from Docker Hub's registry API [FULL]), which fetches the same URL.
   The sha256 is the one OME publishes beside it (``.../7.0.0/artifacts/SHASUMS`` [FULL]).

2. and 3. The DISK and LightGlue weights. ``import valis`` imports ``valis.registration``,
   whose module body builds the default matcher (``registration.py:72``):
   ``feature_detectors.DiskFD()`` calls ``kornia.feature.DISK.from_pretrained('depth')``
   (``feature_detectors.py:578``) and ``feature_matcher.LightGlueMatcher`` builds
   ``kornia.feature.LightGlueMatcher('disk')`` (``feature_matcher.py:1321``). In Kornia
   0.8.3 (the wheel from PyPI: ``kornia/feature/disk/disk.py:145-155``,
   ``kornia/feature/lightglue.py:432, 526-547`` [PARTIAL: those lines]) both go through
   ``torch.hub.load_state_dict_from_url``, which looks in ``$TORCH_HOME/hub/checkpoints``
   and downloads only when the file is absent (``torch/hub.py`` [PARTIAL: get_dir and
   load_state_dict_from_url, read in torch 2.3.1]). The file names below are the ones that
   code computes: the URL's last component for DISK, and
   ``"disk_lightglue_v0.1_arxiv.pth".replace(".", "-")`` for LightGlue. If either name were
   wrong, smoke.sh's import would try to download and fail the build.
   Their sha256 values were computed on 2026-10-10 from files fetched from the two URLs;
   neither upstream publishes a checksum. The DISK URL is a branch, not a release, so the
   check is what notices if it moves.

It also opens up ``<package>/data``: ``init_jvm`` REWRITES ``data/bf_formats.txt`` inside
the installed package every time it starts the JVM (``slide_io.py:95, 274-279``; new in
1.2.0). As another user, or in a read-only Apptainer image, that write fails. The
permissions here cover the first case; the second needs ``--writable-tmpfs`` from whoever
runs the image (containers/README.md says so in this image's row).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
import urllib.request
from pathlib import Path

BIOFORMATS_VERSION = "7.0.0"
JAR = (
    "https://downloads.openmicroscopy.org/bio-formats/"
    f"{BIOFORMATS_VERSION}/artifacts/bioformats_package.jar",
    "a38e1c9efb91f134591fe2c86d731b84f54765a99883f4120ffedea913a121a9",
    "bioformats_package.jar",
)
WEIGHTS = (
    (
        "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth",
        "9c2ee4ded238892dfa51569941372601e35e4a74aa6f84ea80053d2ab1c07abe",
        "depth-save.pth",
    ),
    (
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth",
        "b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335",
        "disk_lightglue_v0-1_arxiv-pth",
    ),
)
READ_ALL = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
WRITE_ALL = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
ENTER_ALL = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH


def package_dir():
    """Where pip put ``valis``, found without importing it (the import is what downloads)."""
    spec = importlib.util.find_spec("valis")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("valis is not installed in this interpreter")
    return Path(list(spec.submodule_search_locations)[0])


def checkpoints_dir():
    home = os.environ.get("TORCH_HOME")
    if not home:
        raise SystemExit(
            "TORCH_HOME is not set: the weights would land in the build user's home, "
            "which is not the home of whoever runs the image"
        )
    return Path(home) / "hub" / "checkpoints"


def fetch(url, sha256, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(url, timeout=300) as response,
        open(partial, "wb") as out,
    ):
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        partial.unlink()
        raise SystemExit(
            f"{url}\n  sha256 {digest.hexdigest()}\n  wanted {sha256}\n"
            "The file behind this URL is not the one this image was written for."
        )
    partial.replace(dest)
    dest.chmod(dest.stat().st_mode | READ_ALL)
    print(f"{dest}  {dest.stat().st_size} bytes  sha256 {sha256}")


def open_up(path, extra):
    path.chmod(path.stat().st_mode | READ_ALL | extra)


def main():
    package = package_dir()
    checkpoints = checkpoints_dir()  # refuses before anything is fetched
    url, sha256, name = JAR
    fetch(url, sha256, package / name)
    for url, sha256, name in WEIGHTS:
        fetch(url, sha256, checkpoints / name)
    for folder in (
        checkpoints,
        *checkpoints.parents[:2],
    ):  # checkpoints, hub, TORCH_HOME
        open_up(folder, ENTER_ALL)

    data = package / "data"
    formats = data / "bf_formats.txt"
    if not formats.is_file():
        raise SystemExit(
            f"{formats} is missing: valis.slide_io reads it at import (slide_io.py:104-111)"
        )
    open_up(data, ENTER_ALL | WRITE_ALL)
    open_up(formats, WRITE_ALL)
    print(f"{data} and {formats.name}: writable by any user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
