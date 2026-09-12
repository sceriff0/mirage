"""A container must not ship a heavyweight framework the pipeline cannot reach.

`tests/test_no_dead_bin_modules.py` already enforces "a `bin/utils` module nothing imports is
dead code and is deleted, not kept just in case". The same argument applies to a multi-hundred-
megabyte framework baked into an image: nothing selects it, nothing imports it, and it is
carried on every pull, in every task, forever.

Three were found this way:

  * `cellpose==3.1.1.1` in `containers/stardist` and `containers/regqc`. It is not
    merely unimported -- `seg_method`'s schema enum is `["stardist", "instantseg", "cellsam"]`,
    so the pipeline **cannot be asked** to run it. Validation refuses the value.
  * `cupy-cuda12x` and `cucim` in `containers/quantify` and `containers/regqc`.
    Zero references anywhere outside `containers/`: no import, no optional import, no CLI use.

**Scope is deliberately a named list, not "every installed package".** Most entries in a
Dockerfile are legitimately transitive or runtime-only dependencies that no first-party file
imports, so a blanket rule would produce noise and get disabled. The list below is the set of
heavyweight, *selectable* frameworks -- the ones whose presence implies a capability. Add to it
when a new framework is introduced.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# distribution name (as written in a Dockerfile) -> python module name to look for
FRAMEWORKS = {
    "cellpose": "cellpose",
    "cupy-cuda12x": "cupy",
    "cucim": "cucim",
    "mesmer": "mesmer",
    "deepcell": "deepcell",
    # containers/tiled installs these for STARE's DISK+LightGlue COARSE front-end
    # (packages/stare/src/stare/coarse_align.py::_frontend_disk_lightglue). They used to live in a separate
    # containers/stare-ml image behind `-profile stare_ml`; that image was never published, so
    # both it and the profile are gone and :tiled carries torch+kornia itself. Both are
    # reachable (imported inside that function, guarded by try/except ImportError), so listing
    # them here does not flag an offender -- it just keeps this guard honest about every
    # heavyweight, selectable framework a container carries, per this file's own instruction to
    # extend the list on a new one.
    "torch": "torch",
    "kornia": "kornia",
}

# Frameworks the pipeline genuinely selects. Each must be reachable: named in seg_method's enum,
# or imported by first-party code.
CODE_DIRS = (
    "bin",
    "packages",  # packages/stare: the STARE method; bin/tiled_*.py are shims over it
    "tests",
    "benchmarks",
    "lib",
    "modules",
    "subworkflows",
    "workflows",
)


def _dockerfiles():
    files = sorted((REPO / "containers").rglob("Dockerfile*"))
    assert files, "found no Dockerfiles -- the glob is wrong, not the repo"
    return files


def _installed(dockerfile_text):
    """Distribution names from FRAMEWORKS that this Dockerfile installs."""
    found = set()
    for dist in FRAMEWORKS:
        # match the dist name as a whole pip token (optionally version-pinned)
        if re.search(
            rf"(?<![\w.-]){re.escape(dist)}(?:==[^\s\\]+)?(?=[\s\\]|$)", dockerfile_text
        ):
            found.add(dist)
    return found


def _seg_method_enum():
    schema = json.loads((REPO / "nextflow_schema.json").read_text())
    out = []

    def walk(node):
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict) and "seg_method" in props:
            out.extend(props["seg_method"].get("enum") or [])
        for v in node.values():
            walk(v)

    walk(schema)
    return set(out)


def _module_is_imported(module):
    pattern = re.compile(rf"^\s*(?:import\s+{module}\b|from\s+{module}\b)", re.M)
    for d in CODE_DIRS:
        root = REPO / d
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            if pattern.search(f.read_text(errors="ignore")):
                return True
    return False


def test_no_container_installs_a_framework_the_pipeline_cannot_reach():
    selectable = _seg_method_enum()
    offenders = []

    for df in _dockerfiles():
        for dist in _installed(df.read_text()):
            module = FRAMEWORKS[dist]
            if module in selectable or dist in selectable:
                continue
            if _module_is_imported(module):
                continue
            offenders.append(f"{df.relative_to(REPO)} installs {dist!r}")

    assert not offenders, (
        "these frameworks are installed in an image but cannot be selected by seg_method and "
        "are imported by nothing -- carried on every pull and every task for no reason:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_the_dockerfile_scan_covers_the_images_this_guard_exists_for():
    """A glob that misses the interesting files passes vacuously."""
    scanned = {f.relative_to(REPO).as_posix() for f in _dockerfiles()}

    for expected in (
        "containers/stardist/Dockerfile",
        "containers/quantify/Dockerfile",
        "containers/regqc/Dockerfile",
    ):
        assert expected in scanned


def test_the_detector_recognises_a_pinned_and_an_unpinned_install():
    """Pin the parser against both forms actually removed, so it cannot rot into a no-op."""
    pinned = "RUN pip install --no-cache-dir \\\n    stardist==0.9.1 \\\n    cellpose==3.1.1.1\n"
    unpinned = "RUN pip install --no-cache-dir \\\n    cupy-cuda12x \\\n    cucim \\\n    numpy\n"

    assert _installed(pinned) == {"cellpose"}
    assert _installed(unpinned) == {"cupy-cuda12x", "cucim"}


def test_the_detector_does_not_fire_on_a_substring():
    """`cucim` must not be matched inside an unrelated token."""
    assert _installed("RUN pip install not-cucim-really\n") == set()


def test_a_selectable_framework_would_be_allowed():
    """The rule must permit what the pipeline can actually run, or it is just a denylist."""
    assert "stardist" in _seg_method_enum() or _module_is_imported("stardist")
