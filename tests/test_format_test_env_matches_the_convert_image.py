"""The format suite must run the reader stack CONVERT_IMAGE actually runs.

requirements/format-tests.txt is a second place a version can be written, which
is exactly what requirements/constraints.txt exists to prevent -- but it cannot
be a constraint file, because containers/convert is a documented exception to
the harmonised set (see that file's header, and
test_container_harmonisation.PINNED_EXCEPTIONS). The next best property is
checkable and checked here: every package named in BOTH files is pinned to the
SAME version, and the packages that make the suite meaningful are present.

The Dockerfile is parsed with test_container_harmonisation's own parser.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORMAT_REQS = ROOT / "requirements" / "format-tests.txt"
DOCKERFILE = ROOT / "containers" / "convert" / "Dockerfile"

# h5py is also pinned by hand in requirements/ci.txt and requirements/testdata.txt (the
# fixture generator writes an HDF5 slide). It cannot go through constraints.txt, which
# only holds packages more than one image installs, so each copy is checked here.
OTHER_H5PY_PINS = {
    "requirements/ci.txt": ROOT / "requirements" / "ci.txt",
    "requirements/testdata.txt": ROOT / "requirements" / "testdata.txt",
}

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
harmonisation = importlib.import_module("test_container_harmonisation")

# The packages without which the format suite proves nothing: the OME-TIFF
# plugin (channel names + pixel size), the generic TIFF plugin, the dispatcher,
# the HDF5 reader, and the two libraries whose versions the readers are pinned
# against.
REQUIRED = {"bioio", "bioio-ome-tiff", "bioio-tifffile", "h5py", "tifffile", "numpy"}

# Installed by the image, deliberately NOT by the format suite. bioio-bioformats
# needs a JVM and jars baked at build time; the three vendor plugins have no
# fixture to read (RULING R3 -- the vendor formats are validated on the cluster).
NOT_IN_THE_SUITE = {"bioio-nd2", "bioio-czi", "bioio-lif", "bioio-bioformats"}


def _pins(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        parsed = harmonisation._parse_req(line)
        if parsed and parsed[1] and parsed[1][0][0] == "==":
            out[parsed[0]] = parsed[1][0][1]
    return out


def format_test_pins() -> dict[str, str]:
    assert FORMAT_REQS.is_file(), f"{FORMAT_REQS} is missing"
    return _pins(FORMAT_REQS.read_text())


def convert_image_pins() -> dict[str, str]:
    tokens = harmonisation._dockerfile_pip_tokens(DOCKERFILE.read_text())
    out: dict[str, str] = {}
    for tok in tokens:
        parsed = harmonisation._parse_req(tok)
        if parsed and parsed[1] and parsed[1][0][0] == "==":
            out[parsed[0]] = parsed[1][0][1]
    assert out, "parsed no pinned package out of containers/convert/Dockerfile"
    return out


def test_every_shared_package_is_pinned_to_the_same_version():
    mine, theirs = format_test_pins(), convert_image_pins()
    drift = {
        name: (mine[name], theirs[name])
        for name in sorted(set(mine) & set(theirs))
        if mine[name] != theirs[name]
    }
    assert not drift, (
        "requirements/format-tests.txt disagrees with containers/convert/Dockerfile "
        f"(package: format-tests -> Dockerfile): {drift}. The IMAGE is the authority: "
        "the format suite exists to run the reader stack CONVERT_IMAGE runs, so a "
        "divergence makes it test something that never ships."
    )


@pytest.mark.parametrize("label", sorted(OTHER_H5PY_PINS), ids=sorted(OTHER_H5PY_PINS))
def test_h5py_stays_in_step_everywhere_else_it_is_pinned_by_hand(label):
    """requirements/ci.txt and requirements/testdata.txt pin containers/convert's
    h5py by hand too."""
    reqs_path = OTHER_H5PY_PINS[label]
    assert reqs_path.is_file(), f"{reqs_path} is missing"
    mine = _pins(reqs_path.read_text())
    theirs = convert_image_pins()
    assert "h5py" in mine, f"{label} does not pin h5py at all"
    assert "h5py" in theirs, "containers/convert/Dockerfile no longer pins h5py"
    assert mine["h5py"] == theirs["h5py"], (
        f"{label} pins h5py=={mine['h5py']}, but containers/convert/Dockerfile pins "
        f"h5py=={theirs['h5py']}. The image is the authority -- see this file's header."
    )


def test_the_overlap_is_not_empty():
    """A negative rule over an empty intersection would pass over nothing."""
    shared = set(format_test_pins()) & set(convert_image_pins())
    assert len(shared) >= 6, (
        f"only {sorted(shared)} appear in both files. Either the Dockerfile moved its "
        "pins somewhere the harmonisation parser cannot see, or the format suite "
        "stopped installing the reader stack -- fix that, not this number."
    )


@pytest.mark.parametrize("package", sorted(REQUIRED))
def test_the_suite_installs_what_makes_it_meaningful(package):
    assert package in format_test_pins(), (
        f"requirements/format-tests.txt does not install {package}. Without it the "
        "format suite either cannot run or runs against a different reader than "
        "production: measured 2026-09-02, an OME-TIFF read without bioio-ome-tiff "
        "reports channel names 'Channel:0:0...' and a pixel size of 1.0."
    )


@pytest.mark.parametrize("package", sorted(NOT_IN_THE_SUITE))
def test_the_jvm_and_vendor_plugins_stay_out_of_the_suite(package):
    """Not a style rule. bioio-bioformats needs a JVM and jars baked into the
    image, and the three vendor plugins have no fixture to read (RULING R3), so
    installing them here buys a slower job and a new failure mode."""
    assert package not in format_test_pins(), (
        f"{package} is installed by the format suite. RULING R3 means there is no "
        "committable fixture for it; the vendor formats are validated on the cluster "
        "(tests/cluster/README.md) and the report is committed as "
        "docs/validation/format_validation.md."
    )


def test_nothing_in_the_format_env_is_unpinned():
    text = FORMAT_REQS.read_text()
    loose = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-") or line == "pytest":
            continue
        parsed = harmonisation._parse_req(line)
        if not parsed or not parsed[1] or parsed[1][0][0] != "==":
            loose.append(line)
    assert not loose, (
        f"unpinned requirement(s) in requirements/format-tests.txt: {loose}. This file "
        "reproduces an image's resolve; an unpinned line makes it a different resolve "
        "every run. (pytest is the one exception -- no requirements/ file pins it.)"
    )
