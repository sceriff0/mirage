"""VALIS 1.2.0 with ONE line of its source changed: a SENSITIVITY row, not VALIS as shipped.

Run it with ``--label valis120_fixed`` (upstream's default call, as ``valis120``) or
``--label valis120_fixed_hires --opt hires=1`` (upstream's high-resolution example, as
``valis120_hires``). Everything else is ``methods/valis.py``, called unchanged: the same
call, the same images, the same writer, the same point path. ``valis120`` and
``valis120_hires`` stay VALIS as shipped and stay the rows the paper reports; these two
rows answer one question beside them: how much of VALIS 1.2.0's result is the defect below.

**The defect** (VALIS 1.2.0 ``registration.py``, ``Valis.prep_images_for_large_non_rigid_
registration``, read in the PyPI sdist ``valis-wsi==1.2.0`` ``[PARTIAL: lines 3800-4089
and register_micro's call at :4899-4905 and field update at :4942-4985; never executed
upstream]``). The method prepares every slide's image for the micro pass. Its first loop
sets ``dxdy = slide_obj.bk_dxdy`` for each slide in turn (``:3965-3968``) and, on the
default path (no tiler, ``norm_method`` set, a non-RGB registrar), does not warp there.
The later normalisation loop (``:4045-4062``) warps each slide with its own ``M`` but with
``bk_dxdy=dxdy``: the value the first loop left, the LAST slide of ``slide_dict``. Every
slide's micro-pass image, the reference's included, is thus prepared with the last
slide's field. ``register_micro`` (``updating_non_rigid=True``, ``:4904``) is the only
caller on which the two differ: the first non-rigid pass passes ``False`` and ``dxdy`` is
None for every slide. The micro correction is then ADDED to each slide's own field
(``:4981-4982``), so the stored transform stays self-consistent and the selfcheck cannot
see the defect. VALIS 1.0.0 warps inside the first loop and is not affected.

**The fix, the whole of it**: one statement inserted after ``slide_obj =
self.get_slide(img_f_list[i])`` in the normalisation loop, re-deriving ``dxdy`` exactly as
the first loop does: ``dxdy = slide_obj.bk_dxdy if updating_non_rigid else None``, plus
one line that records which slides it ran for (``info.micro_fix``). The second issue of
``methods/valis.py`` (the ``shape[1:2]`` test of ``:675``) is NOT changed: one defect, one
row, so a difference between the two rows is that defect's.

**How it is applied.** The method's source is read from the INSTALLED ``registration.py``
(``method_source``: the class's ``FunctionDef`` by ``ast``, dedented), checked against the
sha256 of the sdist's (``SOURCE_SHA256``; the vendored copy is
``benchmarks/tests/regbench/fixtures/valis120_prep_images.txt``), patched as text
(``patched_source``, which refuses unless the anchor and the stale use are each found
exactly where the sdist has them), compiled against a copy of the module's globals, and
set on ``registration.Valis`` for the length of the run only (``installed``). Nothing in
the image changes: the row runs in ``REGBENCH_SIF_VALIS120``. A source of other bytes is
refused (``any_version=1`` lets it through when the anchors are still found, and the
record says so).

``python -m benchmarks.regbench.methods.valis_fixed --check`` run INSIDE the image applies
the patch to the installed VALIS without registering anything: a launch-time check.

Families: the rows are the family ``valis120_fixed`` (``variants.FAMILY``), with no
primary row, never a candidate for ``valis120``'s best and never in the headline tables;
the paper's ``valis120_fix`` table pairs each with its shipped row (``SHIPPED_ROW``).

This module and ``methods/valis.py`` are separate methods so that ``valis``'s code
fingerprint (``methods.code_fingerprint``; ``method_code_freeze.json``) does not move:
adding these rows leaves every ``valis120`` result reusable.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import os
import textwrap
from pathlib import Path

from . import _valis_points as vp
from . import require_fluorescence
from . import valis as _valis

SCOPE = _valis.SCOPE
VARIANTS = ("valis_fixed_rigid", "valis_fixed", "valis_fixed_micro")
LABEL = "valis120_fixed"
HIRES_LABEL = "valis120_fixed_hires"
VARIANT_LABELS = {
    HIRES_LABEL: (
        "valis120_fixed_hires_rigid",
        "valis120_fixed_hires_nonrigid",
        "valis120_fixed_hires",
    )
}
# each fixed run -> the run of VALIS as shipped it is the sensitivity row of
SHIPPED_ROW = {LABEL: "valis120", HIRES_LABEL: _valis.HIRES_LABEL}
# the inner adapter's result names -> this method's
_RENAME = dict(zip(_valis.VARIANTS, VARIANTS))
SELFCHECK_SUBPROCESS = _valis.SELFCHECK_SUBPROCESS
VERSION = _valis.VERSION
FIXED_JVM_HEAP = _valis.FIXED_JVM_HEAP
OPTIONS = _valis.OPTIONS

METHOD = "prep_images_for_large_non_rigid_registration"
# sha256 of method_source() over the PyPI sdist valis-wsi==1.2.0 (lines 3800-4089)
SOURCE_SHA256 = "478c02cd95dcdfa5d71c9a40ce5d89efc29ea8b5025aa4b79aa09cb650dafe7c"
ANCHOR = "slide_obj = self.get_slide(img_f_list[i])"  # the normalisation loop's lookup
FIRST_LOOP = "dxdy = slide_obj.bk_dxdy"  # the first loop's assignment, the one source of dxdy
STALE_USE = "bk_dxdy=dxdy,"  # the warp the anchor feeds
RECORD = "_REGBENCH_FIXED"  # the name the inserted line appends to, in the copied globals
FIX = (
    "dxdy = slide_obj.bk_dxdy if updating_non_rigid else None"
    "  # regbench valis_fixed: this slide's own field",
    f"{RECORD}.append((bool(updating_non_rigid), str(slide_obj.name), dxdy is not None))",
)
NOTE = (
    "ONE line of VALIS 1.2.0 changed (methods/valis_fixed.py): in "
    f"Valis.{METHOD}'s normalisation loop (registration.py:4045-4062) each slide is "
    "warped with its OWN bk_dxdy, re-derived as the first loop derives it, instead of "
    "the last slide's; set on registration.Valis for the run only, from the installed "
    "source checked against the sdist's sha256"
)


def method_source(path):
    """The source of ``Valis.<METHOD>`` in the file ``path``: its lines, dedented."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Valis"), None
    )
    fn = next(
        (n for n in (cls.body if cls else ()) if isinstance(n, ast.FunctionDef)
         and n.name == METHOD),
        None,
    )
    if fn is None:
        raise RuntimeError(f"{path}: no Valis.{METHOD}")
    if fn.decorator_list:
        raise RuntimeError(f"{path}: Valis.{METHOD} is decorated; this patch is not for it")
    lines = text.splitlines(keepends=True)[fn.lineno - 1 : fn.end_lineno]
    return textwrap.dedent("".join(lines))


def sha256(src):
    return hashlib.sha256(src.encode()).hexdigest()


def patched_source(src):
    """``src`` with ``FIX`` inserted after the normalisation loop's ``ANCHOR``. Refuses a
    source where the anchor, the first loop's assignment or the stale use is not found
    exactly once, where the sdist has them: the line is then not the line that was read."""
    lines = src.splitlines(keepends=True)
    at = [i for i, line in enumerate(lines) if line.strip() == ANCHOR]
    first = [i for i, line in enumerate(lines) if line.strip() == FIRST_LOOP]
    if len(at) != 1 or len(first) != 1 or not first[0] < at[0]:
        raise RuntimeError(
            f"Valis.{METHOD}: expected one {ANCHOR!r} after one {FIRST_LOOP!r}, found "
            f"{len(at)} and {len(first)}; this is not the source the fix was written for"
        )
    i = at[0]
    after = [line.strip() for line in lines[i + 1 : i + 4]]
    if not after or not after[0].startswith("processed_warped_img = warp_tools.warp_img(") \
            or STALE_USE not in after:
        raise RuntimeError(
            f"Valis.{METHOD}: the line after {ANCHOR!r} is not the warp with {STALE_USE!r}"
        )
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    return "".join([*lines[: i + 1], *(f"{indent}{s}\n" for s in FIX), *lines[i + 1 :]])


def build(registration, any_version=False):
    """``(function, record, how)``: the patched method compiled against a COPY of
    ``registration``'s globals (the module itself is not touched), the list its inserted
    line appends ``(updating_non_rigid, slide name, has a field)`` to, and what was
    read."""
    path = inspect.getsourcefile(registration)
    src = method_source(path)
    digest = sha256(src)
    if digest != SOURCE_SHA256 and not any_version:
        raise RuntimeError(
            f"{path}: Valis.{METHOD} is not the source of valis-wsi 1.2.0's sdist (sha256 "
            f"{digest}, expected {SOURCE_SHA256}); the fix is refused (any_version=1 "
            "applies it where its anchors are still found)"
        )
    record = []
    scope = dict(vars(registration))
    scope[RECORD] = record
    names = {}
    exec(compile(patched_source(src), f"<valis_fixed: {path}>", "exec"), scope, names)
    how = {
        "source": str(path),
        "source_sha256": digest,
        "source_is_sdist": digest == SOURCE_SHA256,
    }
    return names[METHOD], record, how


@contextlib.contextmanager
def installed(registration, any_version=False):
    """The patched method on ``registration.Valis`` inside the block, the shipped one
    after it (a run is its own process; the restore is for the tests and the selfcheck)."""
    fn, record, how = build(registration, any_version)
    shipped = registration.Valis.__dict__[METHOD]
    setattr(registration.Valis, METHOD, fn)
    try:
        yield record, how
    finally:
        setattr(registration.Valis, METHOD, shipped)


def micro_fix(record, how, last_slide, stage):
    """``info.micro_fix`` of one result. ``stage`` is ``before_micro`` (the rigid and
    non-rigid rows: VALIS as shipped, the fix cannot reach them) or ``micro``."""
    micro = [(name, field) for updating, name, field in record if updating]
    ran = bool(micro)
    out = {
        **how,
        "applied": True,
        "line": FIX[0].split("  #", 1)[0],
        "affects_this_row": stage == "micro" and ran,
        "normalisation_loop_ran": ran if stage == "micro" else None,
        "last_slide": last_slide,
    }
    if stage == "micro":
        # the slides whose micro-pass image differs from the shipped one: all but the
        # last slide, whose own field IS the one the shipped loop reused
        out["slides_given_own_field"] = [n for n, _ in micro if n != last_slide]
        if not ran:
            out["why_no_effect"] = (
                "the normalisation loop did not run (the tiler engaged, norm_method None "
                "or an RGB registrar): this row is VALIS as shipped"
            )
    return out


def _implementation(impl):
    impl = dict(impl or {})
    impl["parameters"] = f"{impl.get('parameters', 'upstream defaults')}, {NOTE}"
    impl["changed_from_defaults"] = {
        **dict(impl.get("changed_from_defaults") or {}),
        f"Valis.{METHOD}": "the normalisation loop warps each slide with its own field "
        "(methods/valis_fixed.py)",
    }
    impl["benchmark_side"] = [*list(impl.get("benchmark_side") or ()), NOTE]
    return impl


def register_group(cases, work, opts):
    work = Path(work)
    label = opts.get("label")
    hires = vp.truthy(opts.get("hires", 0))
    if label is not None and (label == HIRES_LABEL) != hires:
        raise ValueError(
            f"the label {HIRES_LABEL} and the option hires=1 name the same run and must "
            f"come together (label {label!r}, hires={int(hires)})"
        )
    for case in cases:
        require_fluorescence(case, "valis")  # before valis is imported, as valis.py
    os.environ.setdefault("MPLCONFIGDIR", str(work / "mplconfig"))
    import valis
    from valis import registration

    any_version = vp.truthy(opts.get("any_version", 0))
    version = str(getattr(valis, "__version__", "?"))
    if version != VERSION and not any_version:
        raise RuntimeError(
            f"this adapter is VALIS {VERSION} with one line changed; the installed valis "
            f"is {version}"
        )
    inner = dict(opts)
    if label is not None:
        # the shipped run's own label, so methods/valis.py's label check holds as written
        inner["label"] = SHIPPED_ROW.get(label, label)
    native_dir = opts.get("native_dir")
    if native_dir is not None:
        inner["native_dir"] = lambda cid, variant: native_dir(cid, _RENAME[variant])
    with installed(registration, any_version) as (record, how):
        for cid, variant, pmap, info in _valis.register_group(cases, work, inner):
            stage = "micro" if variant == _valis.VARIANTS[-1] else "before_micro"
            own = dict(info)
            own["implementation"] = _implementation(info.get("implementation"))
            own["micro_fix"] = micro_fix(
                record, how, info.get("last_slide_in_slide_dict"), stage
            )
            own["shipped_row"] = SHIPPED_ROW.get(label or "")
            yield cid, _RENAME[variant], pmap, own


def check():
    """Apply the patch to the INSTALLED VALIS and report, without registering anything
    (``python -m benchmarks.regbench.methods.valis_fixed --check`` inside the image)."""
    import valis
    from valis import registration

    with installed(registration) as (_, how):
        patched = getattr(registration.Valis, METHOD)
        ok = patched.__code__.co_filename.startswith("<valis_fixed:")
    restored = getattr(registration.Valis, METHOD)
    ok = ok and not restored.__code__.co_filename.startswith("<valis_fixed:")
    return {"valis": str(getattr(valis, "__version__", "?")), **how, "patched_and_restored": ok}


if __name__ == "__main__":
    import json
    import sys

    if sys.argv[1:] != ["--check"]:
        raise SystemExit("usage: python -m benchmarks.regbench.methods.valis_fixed --check")
    report = check()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["patched_and_restored"] and report["source_is_sdist"] else 1)
