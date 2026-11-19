"""The container images are one dependency surface, and this file is where it is declared.

Checks: every shared package is pinned to one version across images, every image installs
what its own scripts import (a missing import fails at runtime, not at build time), and
requirements files are actually installed.

Version ceilings are set by the base images' Python (mostly 3.10), not by what is newest on
PyPI. ``containers/spatialdata`` is the one exception: it is ``python:3.12-slim`` because
spatialdata>=0.8.0 requires zarr>=3, and it shares no zarr-touching code with the rest.
"""

import ast
import re
import sys
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

from tests.ci_actions import strip_line_comment
from tests.nfmodel import processes, strip_comments
from tests.stare_shims import PACKAGE_SRC, package_module_file

REPO = Path(__file__).resolve().parent.parent
CONTAINERS = REPO / "containers"

# Our own packages, installed from their repositories at a pinned release. The import walker
# follows `from stare.x import y` into the installed package (tests/stare_shims.PACKAGE_SRC)
# instead of reporting `stare` as third-party; what is checked is that an image whose scripts
# reach it pins the distribution in a requirements file it copies.
# {import name: pip distribution name}
FIRST_PARTY_PACKAGES = {
    "stare": "stare-registration",
}
REQUIREMENTS = REPO / "requirements"
CONSTRAINTS = REQUIREMENTS / "constraints.txt"

# The component name in an image reference is the build-context directory name, so no
# translation table is needed.

# The one image allowed to diverge: python:3.12-slim, because spatialdata>=0.8.0 requires
# zarr>=3, which cannot be installed on the python:3.10 bases the others use.
PY312_ISLAND = "spatialdata"

# Packages installed by more than one image must agree on a version. The table is read from
# requirements/constraints.txt, the file the Dockerfiles pass to pip with `-c` and CI installs
# against; the rationale for each pin lives there.


def _read_constraints(path=CONSTRAINTS):
    """{name: version} from a pip constraints file. Exact pins only, which is all it may hold.

    A constraints file may legally carry inexact specifiers; this one may not, which keeps
    `>=`-style drift out of the harmonised set. Comment lines are skipped.
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


# Per-image exceptions to HARMONISED. Each names the upstream constraint that forces it;
# test_every_pinned_exception_is_still_doing_something fails once an exception goes stale.
PINNED_EXCEPTIONS = {
    # bioio-ome-tiff 1.4.0 requires tifffile[zarr]<2025.1.10 on python_version < "3.11"; the
    # convert base (eclipse-temurin:21-jre-jammy) is Python 3.10.
    ("convert", "tifffile"): (
        "2024.12.12",
        "bioio-ome-tiff 1.4.0 caps tifffile<2025.1.10 on py<3.11",
    ),
    # The bioio 3.5.0 plugin set will not resolve against numpy 1.26.4 (bioio-lif's dask chain).
    # Safe here only: this image has no TensorFlow/StarDist, which hold the harmonised numpy at 1.x.
    ("convert", "numpy"): (
        "2.2.6",
        "bioio 3.5.0 plugin set cannot resolve with numpy 1.26.4",
    ),
}

# Packages a container's FROM base image bakes in, so the Dockerfile never pip-installs them.
BASE_IMAGE_PROVIDES = {
    # pytorch/pytorch:*-cuda*-cudnn*-runtime bakes in a CUDA-matched torch; a pip install of a
    # second torch could replace it with a mismatched wheel.
    "cellsam": {"torch"},
    "instanseg": {"torch"},
    # tensorflow/tensorflow:2.15.0-gpu-jupyter bakes in a GPU-matched TensorFlow (same risk as
    # torch above). stardist imports it, and lib/SegBackends.groovy reports its version.
    "stardist": {"tensorflow"},
    # containers/valis pip-installs nothing: it is upstream's digest-pinned cdgatenbee/valis-wsi
    # and its smoke.sh fails the build if pip freeze differs from the base image's. This list is
    # the base image's Python stack (see the containers/valis/Dockerfile header).
    "valis": {
        "valis-wsi",
        "valis",
        "numpy",
        "scipy",
        "scikit-image",
        "pandas",
        "tifffile",
        "psutil",
        "torch",
        "torchvision",
        "kornia",
        "pyvips",
        "scyjava",
        "jpype1",
    },
}

# Packages removed for having no importer, which must not come back. aicsimageio's lazy import
# in the vendored bin/utils/cse/functions.py is in functions nothing calls, so it never runs.
FORBIDDEN = ("cellpose", "cucim", "cupy", "aicsimageio")

# Requirement tokens are parsed with ``packaging``, not a regex, so extras
# (``dask[array]>=2024.8.0``) and multi-clause specifiers (``numpy>=1.26,<3``) are read correctly.
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


# Module-level, so an empty constraints file fails at collection. Defined after _parse_req.
HARMONISED = _read_constraints()


def _dockerfile(name):
    return (CONTAINERS / name / "Dockerfile").read_text()


def _container_dirs():
    return sorted(p.name for p in CONTAINERS.iterdir() if (p / "Dockerfile").is_file())


def _dockerfile_pip_tokens(text):
    """Package tokens from a Dockerfile's pip-install commands only.

    Scoped to ``pip install`` lines, so prose in RUN bodies and echo strings is not read as a
    package. Each ``&&`` segment is searched, so a chained
    ``pip install a && pip install b`` contributes both.
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
                    # `git+https://.../<repo>.git@<rev>` has no requirement token; recover the
                    # package name from the repo segment (e.g. cellsam's cellSAM).
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

    Resolved by the basename the Dockerfile installs, not the container's name, since one
    image may install several requirements files.
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
    # Strip `#` comments so a comment explaining a removal is not read as a reinstall.
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

    SEGMENT resolves its container and entrypoint from ``lib/SegBackends.groovy`` at runtime,
    so the regex scan in ``_module_container_and_scripts`` cannot see them; this parses the
    same table.
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

    Profile-bound container overrides are not scanned: ``nextflow.config`` has none. If one is
    added, scan it with a non-vacuity assertion.

    WARP_SEG_QC is not attributed: its VALIS image is upstream's, and ``warp_seg_qc.py`` is one
    file dispatched by ``--method`` at runtime, so its static import graph includes VALIS-only
    imports that never run under the tiled image.
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


# From the interpreter rather than hand-listed, so no stdlib module reads as a missing package.
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

    A lazy ``import x`` inside a ``def`` is a runtime dependency, declared in
    ``REQUIRED_RUNTIME_IMPORTS`` instead. Class bodies are skipped too (no script in ``bin/``
    has a class-body import). Module-scope ``if``/``try``/``with`` blocks are descended into,
    because they run at import time.
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

    Shared by ``_reachable_local_files`` and ``_third_party_imports`` so name resolution has
    one definition.
    """
    if isinstance(node, ast.Import):
        # `import stare.stages.coarse` -- the dotted name, so the walker can follow it
        # into the installed package; anything else is its top-level distribution name.
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
    # ``from utils.tiled_io import open_lazy``: follow into the submodule only when the head
    # is local, or ``from skimage.transform import warp`` would yield a package "transform".
    if head in local_files or head in local_pkgs:
        if node.module and "." in node.module:
            names.append(node.module.split(".")[-1])
    if node.level:
        # A relative import is always local: follow the module it names, queue imported
        # names that are local modules, and treat the rest as symbols.
        if node.module:
            names.append(node.module.split(".")[-1])
        names += [a.name for a in node.names if a.name in local_files]
    elif head is None or head in local_pkgs:
        names += [a.name for a in node.names]
    return names


def _reachable_local_files(script, root=None):
    """``(files, local_files, local_pkgs)``: every local ``.py`` file reachable from
    ``script`` (``script`` included) by following local imports at any nesting depth.

    ``root`` defaults to ``REPO / "bin"`` and can be overridden to point the walker at a
    scratch directory (see ``test_walker_ignores_imports_nested_in_function_bodies``).

    Returns the ``local_files``/``local_pkgs`` tables alongside ``files`` so a caller that
    needs to resolve further names against them (``_third_party_imports``,
    ``test_required_runtime_imports_are_actually_reached``) does not repeat the ``rglob``.
    """
    root = root or (REPO / "bin")
    local_files = {p.stem: p for p in root.rglob("*.py")}
    # Local packages too: `local_files` is keyed by file stem, so a package directory such as
    # bin/utils/cse/ would otherwise look like a third-party import.
    local_pkgs = {
        d.name for d in root.rglob("*") if d.is_dir() and (d / "__init__.py").is_file()
    }
    seen, queue, files = set(), [script], []
    while queue:
        cur = queue.pop()
        if cur in seen:
            continue
        seen.add(cur)
        # a dotted first-party-package name resolves to its installed file;
        # a bare name is a bin/ module stem
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

    Not a regex, which would match prose in docstrings ("import the module first").
    Module scope only; lazy imports are declared in ``REQUIRED_RUNTIME_IMPORTS``.
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


def _pins_distribution(requirements_text, dist):
    """True when a requirements file pins ``dist`` (``dist @ url``, ``dist==x``, ``dist>=x``)
    on a non-comment line."""
    pat = re.compile(rf"^\s*{re.escape(dist)}\s*(?:@|==|>=|~=|\[)", re.I)
    return any(
        pat.match(ln)
        for ln in requirements_text.splitlines()
        if not ln.lstrip().startswith("#")
    )


def _first_party_packages_installed(container):
    """Import names from FIRST_PARTY_PACKAGES this Dockerfile installs: a
    requirements/<file>.txt it COPYs (and pip-installs with -r, like every pin file) pins
    the package's distribution."""
    text = re.sub(r"\\\s*\n", " ", _dockerfile(container))
    body = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))
    found = set()
    for req in set(re.findall(r"requirements/([A-Za-z0-9._-]+\.txt)", body)):
        path = REQUIREMENTS / req
        if not path.is_file():
            continue
        for name, dist in FIRST_PARTY_PACKAGES.items():
            if _pins_distribution(path.read_text(), dist):
                found.add(name)
    return found


def test_the_first_party_package_walker_follows_the_shim_into_the_package():
    """Non-vacuity for the package-aware walk: bin/tiled_coarse.py is a shim over
    stare.stages.coarse, and following it must reach the package's coarse_align.py --
    the file that holds the COARSE anchor (and whose imports this walk must therefore see)."""
    files, _, _ = _reachable_local_files("tiled_coarse.py")
    names = {p.name for p in files}
    assert "tiled_coarse.py" in names and "coarse.py" in names, sorted(names)
    assert "coarse_align.py" in names, sorted(names)
    assert _first_party_packages_reached("tiled_coarse.py") == {"stare"}
    assert "stare" not in _third_party_imports("tiled_coarse.py")


@pytest.mark.parametrize("container", _container_dirs())
def test_image_whose_scripts_reach_a_first_party_package_installs_it(container):
    """The `stare` twin of the bioio rule: a shim that imports the package at module
    scope fails at import in any image that does not pin stare-registration -- and
    an image that installs it while none of its scripts reach it carries dead weight.
    (A shim importing `stare` fails in any image whose requirements do not pin it.)"""
    reached = set()
    for script in _module_container_and_scripts().get(container, set()):
        reached |= _first_party_packages_reached(script)
    installed = _first_party_packages_installed(container)
    assert reached <= installed, (
        f"containers/{container} runs scripts that import "
        f"{sorted(reached - installed)} at module scope but its Dockerfile does not COPY "
        "and pip-install it (pin its distribution in a requirements file it COPYs; see "
        "FIRST_PARTY_PACKAGES)."
    )
    assert installed <= reached, (
        f"containers/{container} installs first-party package(s) "
        f"{sorted(installed - reached)} that none of its scripts import."
    )


def test_the_first_party_install_scan_reads_the_stare_dockerfile():
    """Non-vacuity: the scan must find the one install that exists, and must NOT be
    satisfied by a commented-out or merely mentioned pin."""
    assert _first_party_packages_installed("stare") == {"stare"}
    assert _pins_distribution(
        "stare-registration @ https://x/v1.tar.gz\n", "stare-registration"
    )
    assert not _pins_distribution(
        "# stare-registration @ https://x\n", "stare-registration"
    )
    assert not _pins_distribution(
        "numpy==1.26.4  # needed by stare-registration\n", "stare-registration"
    )


def test_walker_ignores_imports_nested_in_function_bodies(tmp_path):
    """A lazy ``import x`` inside a ``def`` (or a ``class`` body) is not a module
    requirement.
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


# Lazy imports a container must install although the module-scope walker does not see them,
# because the call reaching them runs on the container's live path.
# {container: {import_name: reason}}. test_required_runtime_imports_are_actually_reached
# fails an entry whose import no longer exists.
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
    "stare": {
        # No torch/kornia: COARSE is an NCC rotation sweep with an ORB fallback
        # (test_tiled_coarse_imports_no_learned_stack).
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
    # Add an entry only after finding the call site on the container's live path.
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

    Walks every local file reachable from the container's scripts with plain ``ast.walk``,
    looking for an import of ``name`` at any depth. An entry whose import no longer exists
    is stale.

    It proves only that the name is imported somewhere reachable, not that the call runs;
    that judgment lives in each entry's reason string.
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

    qc.py's create_registration_qc reaches read_info/read_plane, which reach _open_bioio only
    for a path outside ome_io._TIFFFILE_READABLE, and every path it reads is one of the
    pipeline's own .ome.tif/.ome.tiff intermediates. Both halves are asserted: neither
    bin/utils/qc.py nor bin/generate_registration_qc.py calls a riskier dispatch name, and the
    producers still emit .ome.tif.
    """
    called = {}
    for rel in ("utils/qc.py", "generate_registration_qc.py"):
        path = REPO / "bin" / rel
        found = set()
        for line in path.read_text().splitlines():
            # strip_line_comment: a naive split would truncate a `#` inside a string.
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
    # require_reader would force a specific backend; if either file calls it (or
    # detect_reader), the "always tifffile" reasoning no longer holds.
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

    # Pipeline half: the inputs are always CONVERT_IMAGE's or REGISTER's own output. Read
    # with strip_comments(raw_body), so a comment naming the pattern cannot satisfy it while
    # the string literal is kept.
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


def test_tiled_coarse_imports_no_learned_stack():
    """COARSE's files must not import torch or kornia ANYWHERE -- not even lazily.

    The anchor runs on numpy/scipy/scikit-image alone; a torch/kornia import would tie the
    image back to ~1 GB of wheels it does not need.
    """
    offenders = []
    for rel, path in (
        ("stare/coarse_align.py", PACKAGE_SRC / "stare" / "coarse_align.py"),
        ("stare/stages/coarse.py", PACKAGE_SRC / "stare" / "stages" / "coarse.py"),
        ("bin/utils/coarse_align.py", REPO / "bin/utils/coarse_align.py"),
        ("bin/tiled_coarse.py", REPO / "bin/tiled_coarse.py"),
    ):
        assert path.is_file(), path
        for fn, lineno in _torch_kornia_import_sites(path):
            offenders.append(f"{rel}:{lineno} (in {fn or 'module scope'})")
    assert not offenders, (
        f"torch/kornia is imported by a COARSE file again: {offenders}. The anchor is "
        "numpy/scipy/scikit-image only; see stare/coarse_align.py's module docstring."
    )


def test_the_torch_kornia_import_detector_still_fires(tmp_path):
    """Non-vacuity for the rule above: a nested and a module-scope import are both found."""
    probe = tmp_path / "probe.py"
    probe.write_text("import torch\ndef f():\n    from kornia.feature import DISK\n")
    assert _torch_kornia_import_sites(probe) == [(None, 1), ("f", 3)]


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
