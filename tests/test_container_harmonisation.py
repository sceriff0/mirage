"""The container images are one dependency surface, and this file is where it is declared.

Ten images build the pipeline. Before harmonisation they disagreed with each other and, in three
places, with themselves:

  * ``containers/merge`` installed a bare ``zarr``. A rebuild picks up zarr 3, where the
    ``zarr.hierarchy`` module is REMOVED (zarr's own 3.0 migration guide lists it). The one
    consumer wraps the call in ``except Exception`` and falls back to a whole-array ``imread``,
    so the failure is not a crash -- it is the lazy read silently turning into a full
    materialisation of the largest artifact in the pipeline.
  * ``containers/segmentation/requirements.txt`` was never ``COPY``d and never installed. It is
    a byte-identical copy of ``containers/debug_diffeo/requirements.txt`` (md5 792dc481...), so
    its ``tifffile==2023.4.12`` line read like a pin while pinning nothing.
  * ``containers/quantification`` -- the image behind seven processes, more than any other --
    pinned none of its nine packages.

The rule that catches the worst class of these is ``test_container_installs_what_its_scripts_import``:
``bin/convert_image.py`` has imported ``bioio`` since 2026-01-05, and no container in the repo has
ever installed it, including after the bioformats image was edited five months later. A container
that does not install what its own scripts import is broken at runtime, not at build time.

Version ceilings are set by the base images' Python, not by what is newest on PyPI. Seven of the
ten bases are Python 3.10 (``ubuntu:22.04``, ``*-jammy``, ``cuda-*-ubuntu22.04``), and zarr 3,
tifffile 2026.x and scipy 1.18 all require >=3.12. ``containers/spatialdata`` is a deliberate
exception: it is ``python:3.12-slim``, it is the only image whose dependency (spatialdata>=0.8.0)
*requires* zarr>=3, and it shares no zarr-touching code with the rest.
"""

import ast
import re
import sys
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

from tests.ci_actions import strip_line_comment
from tests.nfmodel import processes, strip_comments
from tests.stare_shims import package_module_file

REPO = Path(__file__).resolve().parent.parent
CONTAINERS = REPO / "containers"

# FIRST-PARTY PACKAGES: repository code that is pip-installed rather than staged onto
# $PATH from bin/. `stare` (packages/stare) is the STARE registration method; the
# bin/tiled_*.py scripts the tiled modules invoke are shims over its stages. The import
# walker follows `from stare.x import y` INTO packages/stare/src (so the lazy torch/kornia
# /zarr/scipy/skimage imports the tiled image's REQUIRED_RUNTIME_IMPORTS entries name are
# still found where they now live), and never reports `stare` as a third-party
# distribution -- what it must check instead is that an image whose scripts reach the
# package INSTALLS it, which `test_image_whose_scripts_reach_a_first_party_package_installs_it`
# does by reading the Dockerfile's COPY + pip install of the package directory.
# {import name: (repo-relative source dir the Dockerfile COPYs, pip distribution name)}
FIRST_PARTY_PACKAGES = {
    "stare": ("packages/stare", "stare-registration"),
}
REQUIREMENTS = REPO / "requirements"
CONSTRAINTS = REQUIREMENTS / "constraints.txt"

# Since the one-repo-per-image rename, the component name in the image reference IS the
# build-context directory name, so no translation table is needed. The previous table mapped
# ten hand-written tags onto differently-spelled directories, which is precisely how
# `instant_seg` and `istantseg` came to misspell InstanSeg two different ways.

# The one image allowed to diverge, and why. It is python:3.12-slim and spatialdata>=0.8.0
# requires zarr>=3, which cannot be installed on the python:3.10 bases the others use.
PY312_ISLAND = "spatialdata"

# Packages installed by more than one image must agree on a version.
#
# THIS TABLE IS NOT WRITTEN HERE ANY MORE. It is READ from requirements/constraints.txt --
# the very file the Dockerfiles COPY and pass to pip with `-c`, and the one CI's requirements
# files constrain against. It used to be a hand-maintained dict beside a hand-maintained copy
# of the same numbers in the workflow YAML, with a 42 KB guard test proving the two copies
# equal. Reading the file deletes the second copy instead of checking it: an image can no
# longer disagree with "the harmonised set" while the harmonised set agrees with nothing that
# actually gets installed.
#
# Ceilings are the newest release that still supports Python 3.10, except numpy, which is held
# at the last 1.x because the segmentation image's TensorFlow 2.15 base requires numpy<2. The
# rationale for each pin, and for the four packages deliberately NOT constrained, lives in
# requirements/constraints.txt itself.


def _read_constraints(path=CONSTRAINTS):
    """{name: version} from a pip constraints file. Exact pins only, which is all it may hold.

    A constraints file may legally carry inexact specifiers; this one may not, and saying so
    HERE is what keeps `>=`-style drift out of the harmonised set. Comment-only lines and the
    documented "not constrained here" block are skipped by the `#` strip, so the block that
    records matplotlib/scikit-learn/opencv/torch's authorities cannot accidentally become a
    pin.
    """
    out = {}
    for raw in path.read_text().splitlines():
        line = strip_line_comment(raw).strip()
        if not line or line.startswith("-"):
            continue
        parsed = _parse_req(line)
        assert parsed, f"{path.name}: cannot parse requirement {line!r}"
        name, specs = parsed
        assert len(specs) == 1 and specs[0][0] == "==", (
            f"{path.name} pins {name!r} as {line!r}. Every line in the constraints file must "
            f"be an exact `name==version`: it is the single version authority for every "
            f"container image and every CI job, and an inexact one authorises drift."
        )
        out[name] = specs[0][1]
    assert out, (
        f"{path} parsed to an EMPTY table. Every harmonised check below would then pass "
        f"having compared nothing -- the vacuous-guard failure this repo keeps hitting."
    )
    return out


# Documented per-image exceptions to HARMONISED. Each names the UPSTREAM CONSTRAINT that forces
# it, because an exception without one is just a silent escape hatch from the harmonised set.
#
# scipy used to be the case that proved the rule: harmonising it to 1.15.3 everywhere made the
# preprocess image unbuildable, since basicpy 1.2.0 -- the in-process illumination-correction
# dependency -- caps scipy below 1.13. That entry is GONE, and its removal is what the exception
# mechanism is for. Illumination correction now runs through nf-core's BASICPY module, in
# labsyspharm's own container; `containers/preprocess` no longer installs basicpy at all, so
# nothing caps its scipy and it is back on the harmonised 1.15.3.
# test_every_pinned_exception_is_still_doing_something is what forced the issue: leaving the
# entry behind would have failed, because the Dockerfile it named no longer diverges.
PINNED_EXCEPTIONS = {
    # bioio-ome-tiff 1.4.0 requires tifffile[zarr]<2025.1.10 on python_version < "3.11"; the
    # convert base (eclipse-temurin:21-jre-jammy) is Python 3.10.
    ("convert", "tifffile"): (
        "2024.12.12",
        "bioio-ome-tiff 1.4.0 caps tifffile<2025.1.10 on py<3.11",
    ),
    # The bioio 3.5.0 plugin set will not resolve against numpy 1.26.4 (bioio-lif's dask chain).
    # Safe HERE only: this image carries no TensorFlow and no StarDist, which are the sole reason
    # the harmonised numpy is held at the last 1.x.
    ("convert", "numpy"): (
        "2.2.6",
        "bioio 3.5.0 plugin set cannot resolve with numpy 1.26.4",
    ),
}

# Packages a container's FROM base image bakes in, so no `pip install` line for them ever
# appears in the Dockerfile -- a script importing one is not missing a dependency, the
# Dockerfile just never had to name it. Documented per-container, each naming the base image
# that provides it, the same way PINNED_EXCEPTIONS documents its upstream constraint.
BASE_IMAGE_PROVIDES = {
    # pytorch/pytorch:*-cuda*-cudnn*-runtime bakes in a CUDA-matched torch build; pip
    # installing a second, unpinned torch here would risk silently replacing that
    # CUDA-matched wheel with a mismatched one.
    "cellsam": {"torch"},
    "instanseg": {"torch"},
    # tensorflow/tensorflow:2.15.0-gpu-jupyter bakes in a GPU-matched TensorFlow build; a
    # pip install of a second one here would risk replacing that wheel with a mismatched
    # one, exactly as for torch below. bin/segment.py does not import tensorflow directly,
    # but stardist 0.9.1 does, and lib/SegBackends.groovy reports its version.
    "stardist": {"tensorflow"},
}

# Packages that are genuinely absent from a given image are fine; these are the ones that must
# never appear again, having been removed as unimported.
#
# aicsimageio went OFF this list when the CSE seg-quality path arrived, then back ON when
# containers/segeval was harmonised. Both moves were right at the time. It is back because the
# only thing it supplied -- a physical pixel size from OME metadata -- is always overridden by
# --pixel-size-um (modules/local/seg_quality_eval.nf always renders it), and because the OME-TIFF
# reader plugin caps tifffile below the harmonised 2025.5.10, making the pair unresolvable.
# bin/utils/cse/functions.py's get_voxel_volume/get_pixel_area still carry a lazy
# `from aicsimageio import AICSImage`, but nothing in bin/ calls either function -- they are dead
# in upstream 1.5.19 and the vendored copy is kept byte-identical -- so that import never runs.
FORBIDDEN = ("cellpose", "cucim", "cupy", "aicsimageio")

# Requirement tokens are parsed with ``packaging`` rather than a regex. A hand-rolled pattern
# missed two real forms already present in this repo -- extras (``dask[array]>=2024.8.0``) and
# multi-clause specifiers (``numpy>=1.26,<3``) -- and silently reported both as "not installed",
# which reads as a missing dependency rather than as a parser gap. ``packaging`` ships with
# pytest, so it is available wherever this suite runs.
_NAME_ONLY = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_.,\-]+\])?$")


def _parse_req(token):
    """(name, [(op, version), ...]) or None if the token is not a requirement."""
    try:
        r = Requirement(token)
    except InvalidRequirement:
        return None
    return r.name.lower(), [(s.operator, s.version) for s in r.specifier]


def _is_exact(specs):
    return len(specs) == 1 and specs[0][0] == "=="


# Module-level, so an unreadable or empty constraints file fails at COLLECTION rather than
# leaving each parametrised check to pass over an empty table. Defined after _parse_req
# because it uses it.
HARMONISED = _read_constraints()


def _dockerfile(name):
    return (CONTAINERS / name / "Dockerfile").read_text()


def _container_dirs():
    return sorted(p.name for p in CONTAINERS.iterdir() if (p / "Dockerfile").is_file())


def _dockerfile_pip_tokens(text):
    """Package tokens from a Dockerfile's pip-install commands only.

    Scoped to ``pip install`` lines on purpose. An earlier version also treated any
    non-directive line as a requirement, which harvested prose from RUN bodies and echo
    strings ("can", "wrapper", "nextflow") as if they were installed packages -- noise that
    could mask a genuinely missing dependency by matching its name.

    Split on ``&&`` BEFORE searching for ``pip install``, and search every resulting segment,
    not just the first. This is load-bearing for containers/tiled, which chains three commands
    on one continuation-joined line (``pip install -r requirements.txt && pip install
    torch==... && pip install kornia==...`` -- the order is required, since kornia drags the
    CUDA torch wheel from PyPI if torch is not already satisfied). Taking only the text up to
    the first ``&&`` silently dropped torch and kornia, which would have made this guard pass
    while torch/kornia were missing from the counted set -- reporting a container "installs
    everything its scripts import" when the packages the DISK+LightGlue front-end actually
    needs were invisible to it. The defect was first found in the now-deleted containers/
    stare-ml image; the same chained form is what containers/tiled uses today.
    """
    joined = re.sub(r"\\\s*\n", " ", text)
    tokens = []
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for segment in stripped.split("&&"):
            segment = segment.strip()
            if not re.search(r"pip3?\s+install", segment):
                continue
            body = re.split(r"pip3?\s+install", segment, maxsplit=1)[1]
            for tok in body.split():
                if tok.startswith("git+"):
                    # `pip install git+https://github.com/<owner>/<repo>.git@<rev>` names no
                    # requirement token at all -- the bare "/" skip below would otherwise drop
                    # it entirely, silently treating a real (pinned-by-commit) install as if the
                    # package were never installed. Recover the package name from the repo path
                    # segment. cellsam's cellSAM is the live example (regqc's cudipy was
                    # removed 2026-08-31 -- its upstream repository no longer exists).
                    m = re.search(r"/([A-Za-z0-9_.\-]+?)(?:\.git)?(?:@.*)?$", tok)
                    if m:
                        tokens.append(m.group(1))
                    continue
                if tok.startswith(("-", "$")) or "/" in tok:
                    continue
                tokens.append(tok.strip('"').strip("'"))
    return tokens


def _requirements_tokens(path):
    tokens = []
    for line in path.read_text().splitlines():
        stripped = strip_line_comment(line).split(";")[0].strip()
        if stripped:
            tokens.append(stripped)
    return tokens


def _requirements_files_installed(text):
    """Every `requirements/<name>.txt` a Dockerfile actually `pip install -r`s.

    The files moved out of ``containers/<c>/requirements.txt`` into ``requirements/<c>.txt``
    so a single ``constraints.txt`` could be shared with CI. This resolves them by the
    BASENAME the Dockerfile installs rather than by the container's own name, because
    containers/tiled installs three (tiled.txt, torch-cpu.txt, kornia.txt) -- and reading only
    ``tiled.txt`` would have hidden torch and kornia from every check below, which is exactly
    the "counted set is smaller than the installed set" defect this module's parser docstring
    describes.
    """
    joined = re.sub(r"\\\s*\n", " ", text)
    found = []
    for m in re.finditer(r"(?:-r|--requirement)[=\s]+(\S+)", joined):
        path = REQUIREMENTS / Path(m.group(1)).name
        if path.is_file():
            found.append(path)
    return found


def _installed(container):
    """package -> specifier string including its operator, or None if installed bare."""
    text = _dockerfile(container)
    tokens = _dockerfile_pip_tokens(text)
    # Only count a requirements file the Dockerfile actually installs.
    for req in _requirements_files_installed(text):
        tokens += _requirements_tokens(req)
    found = {}
    for tok in tokens:
        parsed = _parse_req(tok)
        if not parsed:
            continue
        name, specs = parsed
        if name in ("pip", "setuptools", "wheel", "upgrade"):
            continue
        if name not in found or not found[name]:
            found[name] = specs
    return found


@pytest.mark.parametrize("container", _container_dirs())
def test_every_requirements_file_is_actually_installed(container):
    """A requirements file nothing installs is a decoy that reads like a pin."""
    req = REQUIREMENTS / f"{container}.txt"
    if not req.is_file():
        pytest.skip(f"{container} has no requirements/{container}.txt")
    text = _dockerfile(container)
    assert re.search(rf"COPY[^\n]*requirements/{container}\.txt", text), (
        f"requirements/{container}.txt is never COPYd into containers/{container}, so every "
        f"version it lists is inert. Either install it or delete it."
    )
    assert req in _requirements_files_installed(text), (
        f"requirements/{container}.txt is COPYd but never pip-installed."
    )


@pytest.mark.parametrize("container", _container_dirs())
def test_no_shared_package_is_installed_unpinned(container):
    """An unpinned shared package makes the image non-reproducible and can change major version."""
    installed = _installed(container)
    # The island may use ">=" (it tracks spatialdata's own floor); everyone else needs "==".
    ok = (lambda s: bool(s)) if container == PY312_ISLAND else _is_exact
    unpinned = sorted(p for p, v in installed.items() if p in HARMONISED and not ok(v))
    assert not unpinned, (
        f"containers/{container} installs {unpinned} without an exact version. A rebuild can "
        f"therefore produce a different image than the one that generated published results."
    )


@pytest.mark.parametrize("container", _container_dirs())
def test_shared_packages_match_the_harmonised_version(container):
    """One version per package across every image that is not the py3.12 island."""
    if container == PY312_ISLAND:
        pytest.skip(f"{container} is the documented python:3.12 / zarr>=3 exception")
    installed = _installed(container)
    wrong = {
        p: (v[0][1], HARMONISED[p])
        for p, v in installed.items()
        if p in HARMONISED
        and _is_exact(v)
        and v[0][1] != HARMONISED[p]
        and PINNED_EXCEPTIONS.get((container, p), (None, None))[0] != v[0][1]
    }
    assert not wrong, (
        f"containers/{container} disagrees with the harmonised set "
        f"(package: found -> expected): {wrong}"
    )


@pytest.mark.parametrize("container", _container_dirs())
def test_no_forbidden_package_reappears(container):
    """Frameworks removed for having zero importers must not drift back in."""
    text = _dockerfile(container)
    for req in _requirements_files_installed(text):
        text += "\n" + req.read_text()
    # Strip `#` comments first. This scan is a regex over raw text, so without this a comment
    # EXPLAINING why a package was removed -- quoting the import line it used to have -- reads as
    # a reinstall and fails the guard. That is backwards: it pressures the next person to delete
    # the rationale rather than keep it. Both Dockerfiles and requirements files comment with `#`.
    # Deliberately the naive split, not `strip_line_comment`: a NEGATIVE rule stays comment-visible
    # (CLAUDE.md verification-reality item 7's `test_ci_stack_pinned.py` exception), unlike item 4's swap above.
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    back = sorted(
        {f for f in FORBIDDEN if re.search(rf"(?m)^\s*{f}\b|[\s\\]{f}[=\s\\]", text)}
    )
    assert not back, (
        f"containers/{container} reinstalls {back}, which was removed for having no importer "
        f"anywhere in bin/. If it is genuinely needed now, add the import first."
    )


def _seg_backends_container_scripts():
    """(container dir, {scripts}) for SEGMENT's three backends (modules/local/segment.nf).

    SEGMENT resolves BOTH its container (``SegBackends.container(params.seg_method)``) and
    its entrypoint script (``backend.entrypoint``) dynamically from ``lib/SegBackends.groovy``
    -- never a literal ``bolt3x/mirage-<name>:`` tag or a literal ``<script>.py \\`` line in
    segment.nf itself -- so the regex scan in ``_module_container_and_scripts`` cannot see
    stardist/instanseg/cellsam at all. Parsed straight out of the same table segment.nf reads
    at runtime, not hand-duplicated, so this stays in sync with the table by construction.
    """
    text = (REPO / "lib" / "SegBackends.groovy").read_text()
    out = {}
    for m in re.finditer(
        r"container\s*:\s*'bolt3x/mirage-([a-z0-9-]+):[^']*'.*?entrypoint\s*:\s*'([a-z0-9_]+\.py)'",
        text,
        re.DOTALL,
    ):
        out.setdefault(m.group(1), set()).add(m.group(2))
    return out


def _module_container_and_scripts():
    """(container dir, {scripts}) for every modules/local/*.nf naming a first-party image,
    plus SEGMENT's backend-dispatched images resolved through lib/SegBackends.groovy (see
    ``_seg_backends_container_scripts``).

    There is deliberately no third source for profile-bound container overrides. There used to
    be: ``-profile stare_ml`` re-pointed TILED_COARSE at a second image, and a container
    reachable only through a profile never appears in any ``modules/local/*.nf``, so it needed
    its own scan. That profile is gone -- torch/kornia moved into :tiled itself -- and
    ``nextflow.config`` now contains no ``withName: '...' { container = ... }`` override at
    all, so the scan returned ``{}`` and both it and its helper were dead code that nothing
    would have flagged. If a profile-bound override is ever reintroduced, restore that scan
    WITH a non-vacuity assertion; without one it silently covers nothing.

    WARP_SEG_QC (modules/local/warp_seg_qc.nf) resolves its container the same
    backend-dispatched way, via ``lib/WarpBackends.groovy``, but is deliberately NOT given
    the same treatment here. Its VALIS backend's image (``cdgatenbee/valis-wsi``) is not a
    first-party ``bolt3x/mirage-*`` image and has no ``containers/`` entry to check against;
    its tiled backend's image (``bolt3x/mirage-tiled``) already gets script coverage from
    tiled_coarse.nf / tiled_reg_tile.nf / etc above. Attributing ``warp_seg_qc.py`` itself to
    'tiled' would be unsound the way SEGMENT's attribution is not: SEGMENT has three separate
    per-backend entrypoint FILES (segment.py / segment_instantseg.py / segment_cellsam.py),
    each installed and run only under its own container, so its whole static import graph is
    a fair claim on that container. ``warp_seg_qc.py`` is ONE file dispatched by a `--method`
    flag read at runtime (``_main_tiled`` vs. the VALIS path in ``main``/``write_report``), so
    its static import graph includes VALIS-only deferred imports (``valis``, ``scyjava``) that
    never execute under `--method tiled` -- attributing them to the tiled container would flag
    a false missing dependency, not a real one.
    """
    out = {}
    for nf in sorted((REPO / "modules" / "local").glob("*.nf")):
        text = nf.read_text()
        ref = re.search(r"bolt3x/mirage-([a-z0-9-]+):", text)
        if not ref:
            continue
        cdir = ref.group(1)
        if not (CONTAINERS / cdir / "Dockerfile").is_file():
            continue
        scripts = set(re.findall(r"([a-z0-9_]+\.py)\s*\\", text))
        if scripts:
            out.setdefault(cdir, set()).update(scripts)
    for cdir, scripts in _seg_backends_container_scripts().items():
        if (CONTAINERS / cdir / "Dockerfile").is_file():
            out.setdefault(cdir, set()).update(scripts)
    return out


# Taken from the interpreter rather than hand-listed: a hand-list silently reports stdlib
# modules as missing dependencies (``__future__``, ``statistics``), which is a guard failing
# for the wrong reason -- the exact failure mode this repo has been bitten by before.
_STDLIB_OK = set(sys.stdlib_module_names)

# Import name -> pip distribution name, where they differ.
_IMPORT_TO_DIST = {
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "bioio": "bioio",
    "instanseg": "instanseg-torch",
}


def _module_level_imports(tree):
    """``Import``/``ImportFrom`` nodes reachable WITHOUT descending into a function or
    class body -- i.e. the ones that execute the moment the module is imported.

    A lazy ``import x`` inside a ``def`` is a RUNTIME dependency, not a module one: the
    import statement only executes if something calls the function. A ``class`` body is
    excluded for a different reason -- it DOES execute the moment the module is imported,
    so an import nested in one is a genuine module-scope import in principle -- but no
    script in ``bin/`` has a class-body import, so excluding ``ClassDef`` bodies here costs
    nothing and buys symmetry with ``FunctionDef``.
    ``test_walker_ignores_imports_nested_in_function_bodies`` below asserts the shape this
    function actually implements, class bodies included.
    ``if``/``try``/``with`` at module scope are still descended into -- a
    ``try: import cupy except ImportError`` guard at module scope runs at import time and
    must still count.

    This is what makes ``_third_party_imports`` module-scope-aware. Before this, walking
    with plain ``ast.walk`` visited every node regardless of nesting, so a script that
    merely imported ``bin/utils/ome_io.py`` (whose ``bioio``/``h5py`` imports are lazy,
    INSIDE its reader-dispatch functions) was reported as requiring ``bioio``/``h5py`` --
    the root cause of every one of the 16 entries the old ``UNREACHABLE_IMPORTS``
    allowlist existed to excuse.
    """
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # executes only if called -- a runtime import, not a module one.
        else:
            stack.extend(ast.iter_child_nodes(node))


def _import_names(node, local_files, local_pkgs):
    """Local/third-party names an ``Import`` or ``ImportFrom`` node references, resolved
    against the local file/package tables. ``[]`` for any other node type.

    Shared by ``_reachable_local_files`` (which walks EVERY import, at any nesting depth,
    to find local modules a script transitively pulls in) and ``_third_party_imports``
    (which only ever sees the subset ``_module_level_imports`` yields), so the name
    resolution itself -- relative imports, ``from pkg.sub import x`` following, the
    vendored-package case -- has exactly one definition.
    """
    if isinstance(node, ast.Import):
        # `import stare.stages.coarse` -- the dotted name, so the walker can follow it
        # into packages/stare/src; anything else is its top-level distribution name.
        return [
            a.name if package_module_file(a.name) else a.name.split(".")[0]
            for a in node.names
        ]
    if not isinstance(node, ast.ImportFrom):
        return []
    head = node.module.split(".")[0] if (node.level == 0 and node.module) else None
    if head in FIRST_PARTY_PACKAGES:
        # `from stare.stages import coarse as _impl` / `from stare.slide_io import
        # open_lazy`: the module itself, plus any imported NAME that is a submodule.
        # Everything else imported from it is a symbol, not a dependency.
        names = [node.module]
        names += [
            f"{node.module}.{a.name}"
            for a in node.names
            if package_module_file(f"{node.module}.{a.name}")
        ]
        return names
    names = [head] if head else []
    # ``from utils.tiled_io import open_lazy`` -- follow into the submodule, but ONLY
    # when the head is itself local. Taking the last component unconditionally turned
    # ``from skimage.transform import warp`` into a package named "transform".
    if head in local_files or head in local_pkgs:
        if node.module and "." in node.module:
            names.append(node.module.split(".")[-1])
    if node.level:
        # A RELATIVE import is local by definition -- there is no such thing as a
        # relative third-party import. Follow the module it names, and treat the
        # imported SYMBOLS as symbols: queue the ones that are themselves local
        # modules, and discard the rest rather than reporting them as missing
        # packages. Without this, the vendored `bin/utils/cse` package's own
        # ``from .functions import cell_size_uniformity, ...`` was read as a dozen
        # third-party dependencies that containers/segeval "failed to install".
        if node.module:
            names.append(node.module.split(".")[-1])
        names += [a.name for a in node.names if a.name in local_files]
    elif head is None or head in local_pkgs:
        names += [a.name for a in node.names]
    return names


def _reachable_local_files(script, root=None):
    """``(files, local_files, local_pkgs)``: every local ``.py`` file reachable from
    ``script`` (``script`` included) by following LOCAL imports, discovered with an
    UNRESTRICTED ``ast.walk`` -- a local import is followed regardless of nesting, which
    is the resolution ``_third_party_imports`` has always performed for LOCAL modules.
    This task only restricts THIRD-PARTY name collection to module scope (in
    ``_third_party_imports`` itself); which local files even exist to check is unchanged.

    ``root`` defaults to ``REPO / "bin"`` and can be overridden to point the walker at a
    scratch directory (see ``test_walker_ignores_imports_nested_in_function_bodies``).

    Returns the ``local_files``/``local_pkgs`` tables alongside ``files`` so a caller that
    needs to resolve further names against them (``_third_party_imports``,
    ``test_required_runtime_imports_are_actually_reached``) does not repeat the ``rglob``.
    """
    root = root or (REPO / "bin")
    local_files = {p.stem: p for p in root.rglob("*.py")}
    # Local PACKAGES, not just modules. `local_files` is keyed by file stem, so a
    # package directory is invisible to it -- `bin/utils/cse/__init__.py` has the stem
    # `__init__`, and `from cse import single_method_eval` therefore looked like a
    # third-party import that containers/segeval failed to install. It is vendored
    # source, staged onto $PATH by Nextflow like the rest of bin/, and pip installs
    # nothing for it. `utils` used to be hardcoded here for exactly this reason;
    # deriving the set covers it and every future vendored package alike.
    local_pkgs = {
        d.name for d in root.rglob("*") if d.is_dir() and (d / "__init__.py").is_file()
    }
    seen, queue, files = set(), [script], []
    while queue:
        cur = queue.pop()
        if cur in seen:
            continue
        seen.add(cur)
        # a dotted first-party-package name resolves to its file under packages/*/src;
        # a bare name is a bin/ module stem, as before
        path = package_module_file(cur) or local_files.get(Path(cur).stem)
        if path is None or not path.is_file():
            continue
        files.append(path)
        for node in ast.walk(ast.parse(path.read_text())):
            for n in _import_names(node, local_files, local_pkgs):
                if n in local_files or n in local_pkgs or package_module_file(n):
                    queue.append(n)
    return files, local_files, local_pkgs


def _third_party_imports(script, root=None):
    """Top-level distributions a script imports AT MODULE SCOPE, parsed with ``ast``.

    Deliberately not a regex: ``^\\s*(import|from)\\s+(\\w+)`` also matches prose inside
    docstrings ("import the module first"), which reported a package named ``the`` as a
    missing dependency. Only a real parse distinguishes an import statement from a sentence.

    MODULE SCOPE ONLY (``_module_level_imports``): an import nested inside a ``def``/
    ``class`` is a RUNTIME dependency a script may never exercise, not a module one. A
    container whose scripts reach one only through a lazy import declares it instead in
    ``REQUIRED_RUNTIME_IMPORTS``, with a test (``test_required_runtime_imports_are_actually_
    reached``) that the lazy import still exists, rather than this function over-reporting
    every reachable file's entire ``ast.walk`` as if it always ran.
    """
    files, local_files, local_pkgs = _reachable_local_files(script, root=root)
    third = set()
    for path in files:
        for node in _module_level_imports(ast.parse(path.read_text())):
            for n in _import_names(node, local_files, local_pkgs):
                if n in local_files or n in local_pkgs or package_module_file(n):
                    continue
                elif n not in _STDLIB_OK:
                    third.add(n)
    return third


def _first_party_packages_reached(script):
    """Import names from FIRST_PARTY_PACKAGES that ``script`` reaches at MODULE scope
    (its own or a local file's), i.e. the packages the image running it must install."""
    files, local_files, local_pkgs = _reachable_local_files(script)
    reached = set()
    for path in files:
        for node in _module_level_imports(ast.parse(path.read_text())):
            for n in _import_names(node, local_files, local_pkgs):
                head = n.split(".")[0]
                if head in FIRST_PARTY_PACKAGES and package_module_file(n):
                    reached.add(head)
    return reached


def _first_party_packages_installed(container):
    """Import names from FIRST_PARTY_PACKAGES this Dockerfile COPYs and pip-installs.

    Read off the comment-stripped Dockerfile: a `COPY <src dir> <dest>` of the package's
    repository directory followed by a `pip install ... <dest>` (any flags, the same
    continuation-joined form ``_dockerfile_pip_tokens`` parses). Both halves are
    required; a COPY that is never installed is a stray file, not a dependency.
    """
    text = re.sub(r"\\\s*\n", " ", _dockerfile(container))
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    body = "\n".join(lines)
    found = set()
    for name, (src_dir, _dist) in FIRST_PARTY_PACKAGES.items():
        for m in re.finditer(rf"^\s*COPY\s+{re.escape(src_dir)}/?\s+(\S+)", body, re.M):
            dest = m.group(1).rstrip("/")
            if re.search(
                rf"pip3?\s+install\b[^\n&]*\s{re.escape(dest)}/?(?:\s|$)", body
            ):
                found.add(name)
    return found


def test_the_first_party_package_walker_follows_the_shim_into_the_package():
    """Non-vacuity for the package-aware walk: bin/tiled_coarse.py is a shim over
    stare.stages.coarse, and following it must reach the package's coarse_align.py --
    the file the tiled image's torch/kornia REQUIRED_RUNTIME_IMPORTS entries live in."""
    files, _, _ = _reachable_local_files("tiled_coarse.py")
    names = {p.name for p in files}
    assert "tiled_coarse.py" in names and "coarse.py" in names, sorted(names)
    assert "coarse_align.py" in names, sorted(names)
    assert _first_party_packages_reached("tiled_coarse.py") == {"stare"}
    assert "stare" not in _third_party_imports("tiled_coarse.py")


@pytest.mark.parametrize("container", _container_dirs())
def test_image_whose_scripts_reach_a_first_party_package_installs_it(container):
    """The `stare` twin of the bioio rule: a shim that imports the package at module
    scope fails at import in any image that did not pip-install packages/stare -- and
    an image that installs it while none of its scripts reach it carries dead weight."""
    reached = set()
    for script in _module_container_and_scripts().get(container, set()):
        reached |= _first_party_packages_reached(script)
    installed = _first_party_packages_installed(container)
    assert reached <= installed, (
        f"containers/{container} runs scripts that import "
        f"{sorted(reached - installed)} at module scope but its Dockerfile does not COPY "
        "and pip-install that package directory (see FIRST_PARTY_PACKAGES)."
    )
    assert installed <= reached, (
        f"containers/{container} installs first-party package(s) "
        f"{sorted(installed - reached)} that none of its scripts import."
    )


def test_the_first_party_install_scan_reads_the_tiled_dockerfile():
    """Non-vacuity: the scan must find the one install that exists, and must NOT be
    satisfied by a COPY alone (a probe Dockerfile with the COPY but no pip line)."""
    assert _first_party_packages_installed("tiled") == {"stare"}
    probe = "COPY packages/stare /tmp/stare\nRUN echo no install\n"
    text = re.sub(r"\\\s*\n", " ", probe)
    assert not re.search(r"pip3?\s+install\b[^\n&]*\s/tmp/stare(?:\s|$)", text)


def test_walker_ignores_imports_nested_in_function_bodies(tmp_path):
    """A lazy ``import x`` inside a ``def`` (or a ``class`` body) is NOT a module
    requirement.

    Measured before this change: ``_third_party_imports`` reported ``bioio`` for every
    script that merely imports ``ome_io``, and 16 ``UNREACHABLE_IMPORTS`` entries existed
    only to say so.
    """
    (tmp_path / "probe_script.py").write_text(
        "import numpy\n"
        "def f():\n"
        "    import bioio\n"
        "class C:\n"
        "    import h5py\n"
        "    def m(self):\n"
        "        from skimage import io\n"
    )
    assert _third_party_imports("probe_script.py", root=tmp_path) == {"numpy"}


# The packages a container must install even though the walker (module-scope only, as of
# this task) no longer sees them: a script that runs in that image executes the lazy
# import anyway, because the call reaching it is unconditional on the container's live
# path. {container: {import_name: reason}}. Each entry is proven still genuine by
# ``test_required_runtime_imports_are_actually_reached`` below, which walks the container's
# scripts (and the local modules they import) with an UNRESTRICTED ``ast.walk`` -- the
# traversal ``_third_party_imports`` deliberately no longer performs -- looking for a
# nested import of that exact name. An entry whose import no longer exists anywhere is
# stale and fails there rather than silently permitting a container to skip installing
# something nothing in it needs any more.
#
# This replaces ``UNREACHABLE_IMPORTS``, which inverted the same problem: it named every
# (container, import) pair the OLD unrestricted walker over-reported and asserted each was
# unreachable. Fixing the walker made the whole allowlist redundant for the writer-only
# bioio/h5py cases (ome_io.py's reader-dispatch functions -- _open_bioio, read_info's h5py
# branch -- are simply invisible to a module-scope walk now), but it also means the walker
# no longer POSITIVELY requires bioio/h5py for containers/convert, where the lazy import
# genuinely runs every time CONVERT_IMAGE handles anything but tifffile/HDF5. This table is
# what still catches that, and the small number of other containers whose OWN scripts (read,
# not copied from the old allowlist) execute a lazy import unconditionally on their one job.
REQUIRED_RUNTIME_IMPORTS = {
    "convert": {
        "bioio": (
            "bin/convert_image.py routes every non-tifffile/non-HDF5 format through "
            "ome_io.detect_reader -> require_reader('bioio') -> _open_bioio; the lazy "
            "import runs on every .czi/.nd2/.lif/... slide."
        ),
        "h5py": "ome_io.read_info's HDF5 branch; CONVERT_IMAGE accepts .h5/.hdf5 inputs.",
        "scyjava": (
            "Task 7 review round 1: bin/utils/ome_io.py::_open_bioio and "
            "bin/convert_image.py::read_image both call "
            "jvm_cache.point_jvm_cache_off_readonly_home() (bin/utils/jvm_cache.py, now "
            "reachable via ome_io.py's and convert_image.py's own module-scope import of "
            "it) for the bioio-bioformats route -- that function's own lazy "
            "`from scyjava import config as sjconf` runs unconditionally every time it "
            "is called, on every .svs/.qptiff/.vsi/.scn/.mrxs/.bif/.ims read."
        ),
    },
    "tiled": {
        "torch": (
            "stare/coarse_align.py's estimate_rigid (packages/stare; bin/tiled_coarse.py "
            "is a shim over stare.stages.coarse) calls _frontend_disk_lightglue "
            "unconditionally, which imports torch lazily (confined there by "
            "test_tiled_container_torch_kornia_imports_are_confined_to_disk_lightglue "
            "below). requirements/torch-cpu.txt installs the CPU wheel."
        ),
        "kornia": (
            "same _frontend_disk_lightglue call as torch above (DISK+LightGlue feature "
            "matching); requirements/kornia.txt installs it, deliberately AFTER torch "
            "(see containers/tiled/Dockerfile's ordering note -- kornia drags the CUDA "
            "torch wheel from PyPI otherwise)."
        ),
        "zarr": (
            "stare/slide_io.py's open_lazy (tifffile's aszarr region-read view; the "
            "package's copy of bin/utils/tiled_io.py) is called directly by the coarse, "
            "reg_tile and stitch stages for every streamed tile read."
        ),
        "scipy": (
            "stare/tile_residual.py's residual_displacement -- called from the "
            "reg_tile stage's main flow -- imports scipy.ndimage.gaussian_filter."
        ),
        "skimage": (
            "stare/tile_residual.py's foreground_fraction/residual_displacement "
            "(both called from the reg_tile stage) import skimage.filters/.registration."
        ),
    },
    "cellsam": {
        "cellSAM": (
            "segment_cellsam.py's segmentation call is a lazy "
            "`from cellSAM import cellsam_pipeline` -- the container's whole reason to "
            "exist. Installed via git+https://github.com/vanvalenlab/cellSAM.git."
        ),
        "zarr": (
            "segment_cellsam.py imports extract_dapi_channel from "
            "bin/utils/segment_io.py, whose lazy open_lazy call reads the input image "
            "via tifffile's zarr view."
        ),
    },
    "instanseg": {
        "instanseg": (
            "segment_instantseg.py's segmentation call is a lazy "
            "`from instanseg import InstanSeg` -- the container's whole reason to "
            "exist. Installed as instanseg-torch."
        ),
    },
    "stardist": {
        "zarr": (
            "segment.py imports extract_dapi_channel from bin/utils/segment_io.py, "
            "whose lazy open_lazy call reads the input image via tifffile's zarr view."
        ),
    },
    "merge": {
        "cv2": (
            "bin/merge_channels_pyramid.py's _downsample_plane_f32 (~line 137) imports "
            "cv2 UNCONDITIONALLY and is the load-bearing caller, reached from the "
            "streaming pyramid path; downsample_image's own `import cv2` (which does "
            "tolerate the import failing, falling back to block averaging) is not what "
            "runs there. requirements/merge.txt installs opencv-python deliberately so "
            "that fallback is never exercised."
        ),
        "zarr": (
            "merge_channels_pyramid.py calls open_lazy (bin/utils/tiled_io.py) "
            "directly to stream each channel's pyramid base level."
        ),
    },
    "preprocess": {
        "zarr": (
            "apply_basic_profiles.py, split_multichannel.py, tile_for_basic.py and "
            "generate_preprocess_qc.py each call open_lazy (bin/utils/tiled_io.py) "
            "directly."
        ),
    },
    "regqc": {
        "zarr": (
            "bin/utils/qc.py's create_registration_qc -- generate_registration_qc.py's "
            "core function -- calls open_lazy directly to stream the "
            "reference/registered pair."
        ),
    },
    "segeval": {
        "matplotlib": (
            "the vendored CSE source runs `import matplotlib.pyplot as plt` at the top "
            "of functions.py's foreground_separation() and never touches `plt` -- dead "
            "in upstream CellSegmentationEvaluator 1.5.19, but the import still "
            "EXECUTES, and single_method_eval.py's main evaluation path calls "
            "foreground_separation(). See requirements/segeval.txt's own comment: "
            "without matplotlib the image builds clean and then dies with "
            "ModuleNotFoundError the first time CSE scores a mask."
        ),
    },
    "spatialdata": {
        "spatialdata": (
            "export_spatialdata.py's build_spatialdata assembles the SpatialData "
            "object via a lazy `from spatialdata import SpatialData` -- this "
            "container's whole reason to exist."
        ),
        "anndata": "the AnnData table assembly (build_spatialdata's live path) lazily imports anndata.",
        "geopandas": "the mask-to-shapes conversion lazily imports geopandas (GeoParquet-backed shapes).",
        "shapely": "the same mask-to-shapes conversion lazily imports shapely.geometry.Polygon.",
        "dask": "the pyramid rechunk helpers lazily import dask.array for the lazy chunked pyramid read.",
        "scipy": (
            "the QC-residual -> cell_mask spatial join lazily imports "
            "scipy.spatial.cKDTree, executed before its early-return guard."
        ),
    },
    # Add any other container whose scripts genuinely execute a lazy import at runtime.
    # Derive by reading the scripts, not by copying an old allowlist inverted -- every
    # entry above was confirmed by finding the actual call site on that container's live
    # path, not by assuming a nested import always runs.
}


@pytest.mark.parametrize(
    "container,name",
    sorted(
        (container, name)
        for container, imports in REQUIRED_RUNTIME_IMPORTS.items()
        for name in imports
    ),
)
def test_required_runtime_imports_are_actually_reached(container, name):
    """The premise of every REQUIRED_RUNTIME_IMPORTS entry, checked rather than trusted.

    Walks WITH plain ``ast.walk`` (the traversal ``_third_party_imports`` deliberately no
    longer performs) over every local file reachable from the container's scripts, looking
    for a nested import of ``name``. An entry whose import no longer exists anywhere is
    stale: it would silently let the container drop a dependency nothing in it needs any
    more, and is exactly the shape of exemption-outliving-its-reason this file has
    already shipped once (containers/segeval's aicsimageio, which was NEVER reached --
    the opposite failure this test exists to catch on the other side).

    This proves only that the name is imported SOMEWHERE reachable from the container's
    scripts -- not that the import is nested (a promoted module-scope import still passes),
    and not that the function containing it is ever actually called. The judgment that the
    import genuinely executes on the container's live path lives in the reason string next
    to each entry, not in this test.
    """
    scripts = sorted(_module_container_and_scripts().get(container, set()))
    assert scripts, (
        f"REQUIRED_RUNTIME_IMPORTS names {container!r}, which owns no scripts in "
        "_module_container_and_scripts() -- nothing to check this exemption against."
    )
    all_files, seen_paths, local_files, local_pkgs = [], set(), {}, set()
    for script in scripts:
        files, local_files, local_pkgs = _reachable_local_files(script)
        for path in files:
            if path not in seen_paths:
                seen_paths.add(path)
                all_files.append(path)
    reached = any(
        name in _import_names(node, local_files, local_pkgs)
        for path in all_files
        for node in ast.walk(ast.parse(path.read_text()))
    )
    assert reached, (
        f"REQUIRED_RUNTIME_IMPORTS[{container!r}][{name!r}] names an import that no "
        f"script in {scripts} (or a local module one of them imports) contains any "
        "more -- the exemption has outlived its reason and must be removed."
    )


def test_qc_reader_dispatch_stays_tifffile_only():
    """Why containers/regqc needs no bioio even though bin/utils/qc.py CALLS ome_io's readers.

    The module-scope walker never sees ome_io's lazy `import bioio` (it is inside
    _open_bioio), and REQUIRED_RUNTIME_IMPORTS deliberately lists no bioio for regqc. That
    is only right because of a narrower claim this test pins: qc.py's create_registration_qc
    (plan 08's before/after panel) reaches read_info/read_plane, and those two reach
    _open_bioio ONLY for a path outside ome_io._TIFFFILE_READABLE -- while every path qc.py
    ever calls them with is one of this pipeline's own .ome.tif/.ome.tiff intermediates. Both
    halves are asserted here so a change to either -- qc.py OR generate_registration_qc.py
    calling a riskier dispatch name (require_reader), or the producers drifting off the
    .ome.tif convention -- fails here rather than silently making the omission wrong.

    The dispatch-name scan covers BOTH bin/utils/qc.py and bin/generate_registration_qc.py:
    the CLI script imports qc.py, and nothing else scans it for a dispatch call added to the
    CLI directly.
    """
    called = {}
    for rel in ("utils/qc.py", "generate_registration_qc.py"):
        path = REPO / "bin" / rel
        found = set()
        for line in path.read_text().splitlines():
            # strip_line_comment, not `line.split("#", 1)[0]`: CLAUDE.md's "Verification
            # reality" #7 forbids a private comment-stripping regex, and the naive cut
            # truncates any line whose `#` is inside a string literal.
            code = strip_line_comment(line)
            for fn in ("detect_reader", "require_reader", "read_info", "read_plane"):
                if f"{fn}(" in code:
                    found.add(fn)
        called[rel] = found
    assert called["utils/qc.py"], (
        "qc.py no longer calls any ome_io reader-dispatch function -- regqc's bioio omission "
        "is then covered by the module-scope walker alone, and this test (now vacuous) "
        "should be removed."
    )
    # detect_reader is pure extension-sniffing (no import); require_reader is what a NEW,
    # riskier read path would call to force a specific backend. Neither is what
    # create_registration_qc calls -- if one appears (in EITHER file), the "always tifffile"
    # branch reasoning below no longer applies and the exemption needs re-deriving, not just
    # re-reading.
    for rel, found in called.items():
        assert found <= {"read_info", "read_plane"}, (
            f"bin/{rel} now calls {found - {'read_info', 'read_plane'}}, not just "
            "read_info/read_plane -- the (regqc, bioio) exemption's reasoning (read_info/"
            "read_plane only reach _open_bioio for a non-_TIFFFILE_READABLE suffix) does not "
            "cover detect_reader/require_reader and must be re-derived."
        )

    ome_io_src = (REPO / "bin" / "utils" / "ome_io.py").read_text()
    tifffile_readable_match = re.search(
        r"_TIFFFILE_READABLE\s*=\s*\(([^)]*)\)", ome_io_src
    )
    assert tifffile_readable_match, (
        "ome_io._TIFFFILE_READABLE not found by this guard's regex"
    )
    tifffile_readable = tifffile_readable_match.group(1)
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        assert f'"{suffix}"' in tifffile_readable, (
            f"ome_io._TIFFFILE_READABLE no longer names {suffix!r} -- "
            "CONVERT_IMAGE/REGISTER's own output convention would then route through "
            "_open_bioio, and the (regqc, bioio) exemption is no longer sound."
        )

    # The pipeline-level half of the claim: reference_path/registered_path/native_path are
    # always CONVERT_IMAGE's or REGISTER's own output, never an arbitrary user-supplied
    # format. Read through tests.nfmodel rather than a raw .read_text() needle: a raw needle
    # is satisfied by a COMMENT that merely mentions the pattern -- measured here, a
    # `// historical: emitted path("*.ome.tif")` comment kept a raw check green while the
    # real declaration changed to something else entirely. strip_comments(process.raw_body)
    # removes exactly that comment while keeping the real string literal verbatim (CLAUDE.md's
    # "Verification reality" #5: the blanked `.body`/`.outputs` view is for LOCATING
    # structure and identifiers; a literal inside a quoted string needs the string-preserving
    # view instead).
    nf_processes = processes()
    convert_image = nf_processes["CONVERT_IMAGE"]
    assert '"*.ome.tif"' in strip_comments(convert_image.raw_body), (
        "CONVERT_IMAGE (modules/local/convert_image.nf) no longer emits '*.ome.tif' in real "
        "(non-comment) code -- the (regqc, bioio) exemption assumed CONVERT_IMAGE's output "
        "is always tifffile-readable, and that assumption needs re-checking against "
        "whatever it emits now."
    )
    register = nf_processes["REGISTER"]
    register_clean = strip_comments(register.raw_body)
    assert '"*.ome.tif"' in register_clean and '"*.ome.tiff"' in register_clean, (
        "REGISTER (modules/local/register.nf) no longer stages '*.ome.tif'/'*.ome.tiff' in "
        "real (non-comment) code -- the (regqc, bioio) exemption assumed REGISTER's output "
        "is always tifffile-readable, and that assumption needs re-checking against "
        "whatever it stages now."
    )


def _torch_kornia_import_sites(path):
    """[(enclosing function name or None, lineno), ...] for every ``import torch``/``import
    kornia`` (or ``from torch``/``from kornia`` ...) anywhere in ``path``, tagged with the
    innermost enclosing function -- ``None`` means module scope.
    """
    tree = ast.parse(path.read_text())
    sites = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name.split(".")[0] in ("torch", "kornia"):
                    sites.append((self.stack[-1] if self.stack else None, node.lineno))
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module and node.module.split(".")[0] in ("torch", "kornia"):
                sites.append((self.stack[-1] if self.stack else None, node.lineno))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sites


def test_tiled_container_torch_kornia_imports_are_confined_to_disk_lightglue():
    """torch/kornia must be imported ONLY inside ``_frontend_disk_lightglue``, never at module
    scope.

    This began as the premise behind two exemptions from an older allowlist, back when
    torch/kornia shipped in a separate image and the import inside :tiled was meant to fail.
    :tiled now installs both (and ``REQUIRED_RUNTIME_IMPORTS["tiled"]`` declares them,
    proven reached by ``test_required_runtime_imports_are_actually_reached`` above), but this
    confinement rule survives that change on its OWN reasoning, independent of the walker
    entirely: ``import torch`` at module scope in ``coarse_align.py`` would make the module --
    and therefore ``bin/tiled_coarse.py`` -- UNIMPORTABLE anywhere torch is absent. That is not
    hypothetical. ``coarse_align.py`` is imported by the tiled oracle
    (``stare/pipeline.py``) and by this test suite, and it must keep importing on a
    plain checkout with no ML stack, so that ``estimate_transform_from_matches``,
    ``normalize_intensity``, ``scale_transform_to_full_res`` and every test that does not touch
    DISK keep working. It is also what turns a torch-less environment into an actionable
    RuntimeError at CALL time instead of an ImportError at import time.

    The confinement is what lets ``_disk_models`` take the model classes as ARGUMENTS instead
    of importing them; see the comment above it in coarse_align.py.
    """
    offenders = []
    for rel in (
        # the method's files, where the import actually lives ...
        "packages/stare/src/stare/coarse_align.py",
        "packages/stare/src/stare/stages/coarse.py",
        # ... and the pipeline's shims over them, which must stay import-free
        "bin/utils/coarse_align.py",
        "bin/tiled_coarse.py",
    ):
        for fn, lineno in _torch_kornia_import_sites(REPO / rel):
            if fn != "_frontend_disk_lightglue":
                offenders.append(f"{rel}:{lineno} (in {fn or 'module scope'})")
    assert not offenders, (
        "torch/kornia is imported outside _frontend_disk_lightglue in a script the tiled "
        f'container runs: {offenders}. The ("tiled", "torch")/("tiled", "kornia") '
        "REQUIRED_RUNTIME_IMPORTS entries assume the import is confined there; a second "
        "import site needs its own justification, not a free ride on this one."
    )


@pytest.mark.parametrize(
    "container,scripts", sorted(_module_container_and_scripts().items())
)
def test_container_installs_what_its_scripts_import(container, scripts):
    """The rule that catches bioio: an image must install what its own scripts import.

    A missing dependency here is invisible at build time and fails at gigapixel scale, after the
    scheduler has already granted the task its memory.

    ``_third_party_imports`` only reports MODULE-SCOPE imports; a container's own lazy
    (runtime-only) imports are declared in ``REQUIRED_RUNTIME_IMPORTS`` and unioned in here
    -- attributed to that table rather than to a script, since a lazy import may live in a
    module none of the container's entry scripts import at module scope either (that is
    the whole reason it needs declaring instead of being discovered).
    """
    installed = set(_installed(container)) | BASE_IMAGE_PROVIDES.get(container, set())
    missing = {}
    for script in sorted(scripts):
        for name in sorted(_third_party_imports(script)):
            dist = _IMPORT_TO_DIST.get(name, name).lower()
            if dist not in installed and name.lower() not in installed:
                missing.setdefault(script, []).append(name)
    for name in sorted(REQUIRED_RUNTIME_IMPORTS.get(container, {})):
        dist = _IMPORT_TO_DIST.get(name, name).lower()
        if dist not in installed and name.lower() not in installed:
            missing.setdefault("REQUIRED_RUNTIME_IMPORTS", []).append(name)
    assert not missing, (
        f"containers/{container} does not install everything its scripts import: {missing}. "
        f"The script runs in this image, so the import fails at runtime."
    )


@pytest.mark.parametrize("key", sorted(PINNED_EXCEPTIONS))
def test_every_pinned_exception_is_still_doing_something(key):
    """An exception that matches the harmonised value is stale and should be deleted.

    Without this, an exception silently outlives the upstream cap that justified it and becomes
    a permanent hole in the harmonised set that nobody re-examines.
    """
    container, package = key
    version, reason = PINNED_EXCEPTIONS[key]
    assert package in HARMONISED, (
        f"{package} is not in the harmonised set, so exempting it is meaningless"
    )
    assert version != HARMONISED[package], (
        f"the {container}/{package} exception pins {version}, which is what HARMONISED already "
        f"requires -- the exception is stale and should be removed."
    )
    assert reason.strip(), (
        "every exception must name the upstream constraint that forces it"
    )
    actual = _installed(container).get(package)
    assert _is_exact(actual) and actual[0][1] == version, (
        f"containers/{container} pins {package}={actual}, but the exception documents {version}. "
        f"The exception and the Dockerfile must agree, or the exception is describing fiction."
    )
