"""The STARE image is slim: no torch, no kornia, no learned weights, no libGL.

STARE's COARSE stage (stare/coarse_align.py) needs neither torch nor kornia. This file pins
that from both ends: containers/stare's Dockerfile installs none of it, and the image's
smoke.sh asserts torch is not importable (only a build-time import check can see a
transitive pull).

The Dockerfile reader is comment-blind because the Dockerfile's comments name
torch/kornia/TORCH_HOME.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STARE = REPO / "containers" / "stare"
DOCKERFILE = STARE / "Dockerfile"
SMOKE = STARE / "smoke.sh"


def _code(path, must_contain):
    """``path`` with comment lines removed, so prose can never satisfy (or trip) a guard."""
    code = "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    for needle in must_contain:
        assert needle in code, (
            f"{path.relative_to(REPO)}: the comment stripper returned something without "
            f"{needle!r} -- every absence asserted below would be vacuous"
        )
    return code


def test_the_stare_dockerfile_installs_no_learned_stack():
    code = _code(
        DOCKERFILE, ("FROM python:", "RUN pip install", "requirements/stare.txt")
    )
    for banned in ("torch", "kornia", "TORCH_HOME", "libgl1", "libglib2.0"):
        assert banned not in code, (
            f"containers/stare/Dockerfile mentions {banned!r} outside a comment. The STARE "
            "image is torch-free by design; the learned COARSE front-end it served is gone."
        )
    # and it still installs the method and the pin set it runs on: STARE itself is a pinned
    # release in requirements/stare.txt, never a copy of a repository directory
    assert "requirements/stare.txt" in code
    assert "packages/stare" not in code
    pins = (DOCKERFILE.parents[2] / "requirements" / "stare.txt").read_text()
    assert re.search(
        r"^stare-registration @ https://github\.com/sceriff0/stare/", pins, re.M
    )
    assert "procps" in code, "Nextflow's task-metrics wrapper needs `ps`"


def test_the_smoke_test_asserts_torch_is_not_importable():
    code = _code(SMOKE, ("import stare",))
    loop = re.search(r"for mod in ([^;]+); do(.*?)done", code, re.S)
    assert loop and "torch" in loop.group(1).split(), (
        "containers/stare/smoke.sh no longer checks that torch is absent from the image"
    )
    body = loop.group(2)
    assert "__import__('${mod}')" in body and "exit 1" in body, (
        "the torch-absence check must try the import and FAIL the build when it succeeds"
    )


def test_the_retired_images_are_gone():
    assert not (REPO / "containers" / "tiled").exists(), (
        "containers/tiled was replaced by containers/stare and must be deleted"
    )
    assert not (REPO / "containers" / "stare-ml").exists(), (
        "containers/stare-ml was folded into containers/tiled (itself since replaced by "
        "containers/stare) and must stay deleted"
    )
    dirs = {
        e["dir"] for e in json.loads((REPO / "containers" / "images.json").read_text())
    }
    assert "stare" in dirs and not dirs & {"tiled", "stare-ml"}, sorted(dirs)
