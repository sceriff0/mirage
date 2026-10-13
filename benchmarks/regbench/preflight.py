"""Before anything is submitted: can each image run what it is asked to run?

``python -m benchmarks.regbench preflight --method M [--label L] [--writable DIR ...]``
is run by ``slurm/submit_cohort.sh`` on the login node, INSIDE each image a selected run
uses, started exactly as a job starts it (``regbench_py``), and can be run alone
(``submit_cohort.sh --preflight``). It checks, and prints ONE LINE PER FAILURE:

* that the modules the run needs import: ``numpy``, ``tifffile``, ``zarr``,
  ``imagecodecs`` and ``stare.slide_io`` for every run (the harness reads every image
  through STARE's reader), the method's own package, and what its adapter imports
  beside it (``NEEDS``);
* the version the adapter enforces: the installed package's version is read the way the
  adapter reads it and compared with the adapter's ``VERSION``. The adapter's own check
  is unchanged; this only says, before a cluster cycle is spent, what it will say;
* that ``numpy`` and the method's package resolve INSIDE the image and not under the
  user's own site-packages (``~/.local``): the launcher sets ``PYTHONNOUSERSITE=1`` in
  every image, and this is the check that it arrived;
* that each folder given with ``--writable`` (the results folder, the work folder) is
  visible from inside the image and can be written to.

An interpreter that is missing never gets this far: the launcher reports an image that
did not start or has neither ``python`` nor ``python3``.

Standard library only at import: it must load in every image.
"""

from __future__ import annotations

import importlib
import os
import site
import sys
import tempfile
from pathlib import Path

# every run: the harness's own reads and writes
COMMON = ("numpy", "tifffile", "zarr", "imagecodecs", "stare.slide_io")
# the method's package (the name it is imported by)
PACKAGE = {
    "stare": "stare",
    "valis_mirage": "valis",
    "valis": "valis",
    "ashlar": "ashlar",
    "deeperhistreg": "deeperhistreg",
    "palom": "palom",
}
# what an adapter imports beside its package
NEEDS = {
    "stare": ("stare.cli", "stare.stage_warp", "stare.tile_grid"),
    # the ASHLAR command path of the benchmark imports STARE's manifest and reader
    "ashlar": ("benchmarks.ashlar.original", "benchmarks.ashlar.retile"),
    "deeperhistreg": ("torch", "SimpleITK"),
    "palom": ("psutil",),
}
# the scorer's image (variants.SCORER_IMAGE): prepare, gate, tissue, score
SCORER = ("numpy", "scipy", "skimage", "tifffile", "zarr", "stare.slide_io", "cv2")
# imported by the harness where present, and only reported where absent
OPTIONAL = ("psutil",)


def _where(module):
    return str(getattr(module, "__file__", "") or "")


def _user_sites():
    out = [str(Path.home() / ".local")]
    try:
        out.append(site.getusersitepackages())
    except Exception:  # an interpreter without a user site
        pass
    return [str(Path(p).resolve()) for p in out if p]


def outside_the_image(path):
    """Is ``path`` under the user's own site-packages or ``~/.local``?"""
    if not path:
        return False
    here = str(Path(path).resolve())
    return any(here == u or here.startswith(u + os.sep) for u in _user_sites())


def installed_version(method, package):
    """The installed package's version, read the way the method's adapter reads it."""
    adapter = importlib.import_module(f"benchmarks.regbench.methods.{method}")
    reader = getattr(adapter, "installed_version", None)
    if reader is not None:
        return str(reader(package)), adapter
    return str(getattr(package, "__version__", "?")), adapter


def unreadable_slides(cases_root, datasets):
    """``(notes, failures)`` for the first prepared case of each dataset: can the
    image files it names be read from here. A folder that is not bound into the image
    fails every unit of every run; one case is enough to see it."""
    from .cases import list_cases

    notes, failures = [], []
    for dataset in datasets:
        try:
            cases = list_cases(cases_root, dataset)
        except Exception as exc:
            failures.append(
                f"the {dataset} cases under {cases_root} cannot be listed from "
                f"inside the image: {type(exc).__name__}: {exc} (REGBENCH_BIND?)"
            )
            continue
        if not cases:
            continue
        case = cases[0]
        paths = [
            p
            for p in (
                case.ref_image,
                case.mov_image,
                case.ref_nuclear,
                case.mov_nuclear,
            )
            if p
        ]
        missing = [str(p) for p in paths if not os.access(str(p), os.R_OK)]
        if missing:
            failures.append(
                f"{dataset}/{case.case_id}: its image file(s) cannot be read from "
                f"inside the image: {', '.join(missing)}. Name the slides' folder in "
                "REGBENCH_BIND, with the path the cases were prepared with"
            )
        else:
            notes.append(
                f"readable: the {len(paths)} image file(s) of {dataset}/{case.case_id}"
            )
    return notes, failures


def check(method=None, writable=(), scorer=False, cases=None, datasets=()):
    """``(notes, failures)``: each a list of lines. ``method`` is an adapter's name
    (``METHODS``); ``scorer`` checks the scorer's image instead. ``cases`` and
    ``datasets``: :func:`unreadable_slides`."""
    notes, failures = [], []
    wanted = list(SCORER if scorer else COMMON)
    package_name = None if scorer else PACKAGE.get(method)
    if package_name:
        wanted.append(package_name)
    wanted += [m for m in NEEDS.get(method, ()) if not scorer]
    loaded = {}
    for name in dict.fromkeys(wanted):
        try:
            loaded[name] = importlib.import_module(name)
        except BaseException as exc:  # an import may raise anything (SystemExit too)
            failures.append(f"cannot import {name}: {type(exc).__name__}: {exc}")
    for name in OPTIONAL:
        if name in wanted:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            notes.append(f"{name} is not installed (the monitor reads /proc instead)")
    for name in ("numpy", package_name):
        module = loaded.get(name)
        if module is not None and outside_the_image(_where(module)):
            failures.append(
                f"{name} resolves OUTSIDE the image, under the user's own "
                f"site-packages: {_where(module)} (PYTHONNOUSERSITE is "
                f"{os.environ.get('PYTHONNOUSERSITE', 'not set')!r} here; the launcher "
                "sets it to 1 in every image)"
            )
        elif module is not None:
            notes.append(f"{name} from {_where(module)}")
    if package_name and package_name in loaded:
        try:
            found, adapter = installed_version(method, loaded[package_name])
            enforced = getattr(adapter, "VERSION", None)
            if enforced is not None and found != str(enforced):
                failures.append(
                    f"{package_name} says its version is {found!r}; the adapter "
                    f"(methods/{method}.py) runs {enforced!r} and nothing else, so "
                    "every unit of this run would fail with that message"
                )
            else:
                notes.append(
                    f"{package_name} {found}"
                    + (" (the version the adapter enforces)" if enforced else "")
                )
        except BaseException as exc:
            failures.append(
                f"cannot read the version of {package_name}: "
                f"{type(exc).__name__}: {exc}"
            )
    for folder in writable:
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".preflight-"):
                pass
            notes.append(f"writable: {folder}")
        except OSError as exc:
            failures.append(
                f"{folder} is not visible or not writable from inside the image: "
                f"{type(exc).__name__}: {exc} (REGBENCH_BIND?)"
            )
    if cases and datasets:
        seen, unread = unreadable_slides(cases, datasets)
        notes += seen
        failures += unread
    notes.append(f"python {sys.version.split()[0]} at {sys.executable}")
    return notes, failures


def main(a):
    what = "the scorer" if a.scorer else a.method
    tag = f"[{a.image or 'no image'} {what}]"
    notes, failures = check(
        a.method,
        a.writable or (),
        a.scorer,
        getattr(a, "cases", None),
        getattr(a, "dataset", None) or (),
    )
    for line in notes:
        print(f"preflight {tag} {line}")
    for line in failures:
        print(f"preflight FAIL {tag} {line}", file=sys.stderr)
    print(f"preflight {tag} " + (f"{len(failures)} failure(s)" if failures else "ok"))
    return 1 if failures else 0
