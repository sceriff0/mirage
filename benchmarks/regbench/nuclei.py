"""Bare nuclei and outlines for the cohort's slides: regbench's own StarDist run.

For each NATIVE slide of a ``prepare multiplex`` manifest (``regbench.manifest``) this
runs the pipeline's ``bin/segment.py``, unchanged, with the arguments the pipeline renders
for ``--seg_method stardist`` (:func:`command`), traces the cell mask it writes to the
slide's outlines (``outlines.trace``: ``bin/mask_to_geojson.py`` at
``simplify_tolerance``, into the manifest's ``ref_cells`` / ``mov_cells`` path) and keeps
the nuclei mask as the slide's bare nuclei. One record per slide is left for ``prepare
multiplex --nuclei-dir``::

    <out>/<group>/<slide>/nuclei.json   where the nuclei came from, whether they may be
                                        used, how well they agree with the outlines
    <out>/<group>/<slide>/nuclei.npz    the nuclei as polygons (when usable)
    <out>/<group>/<slide>/rerun/        the masks of the StarDist run of that slide
    <outlines>/<group>/<slide>.geojson  the outlines (+ ``.traced.json``: what they were
                                        traced from, at which tolerance)

The outlines are DEFINED to be what the arms' QC published (``outlines.py``;
``benchmarks/tests/regbench/test_outline_definition.py`` pins every link): the StarDist
nuclei grown by ``seg_expand_distance`` and traced by the pipeline's own function. The
nuclei are then by construction the ones the outlines were grown from, and the comparison
below is a SELF-CONSISTENCY check: a mismatch is a tracing bug, not another segmentation.
``exact`` must come out true; false is printed loudly and kept.

A manifest written from the arms' results before 2026-10-16 names the arm's PUBLISHED
outlines; those are never overwritten (a GeoJSON without the ``.traced.json`` beside it is
not this module's), and the StarDist run is compared with them as it always was
(``outlines_from: "manifest"``).

**A record is of one image.** It holds the fingerprint of the slide it was computed from
(``image_fingerprint``: size, modification time and sampled bytes,
``cases.file_fingerprint``). A record whose fingerprint is not the file's now, or that has
none, is not used: the slide is decided again, and a run kept for it is set aside
(``rerun_stale/``, its traced outlines with it) and run again. ``--force``
(``REGBENCH_NUCLEI_FORCE=1``) does that for EVERY slide, whatever its record says.

A run under ``ACCEPT_ONE_NUCLEUS`` is REFUSED (user ruling 2026-10-11): the slide has no
nuclei and its record says why; between that and the exact thresholds it is kept as
``exact: false``. Anything else (shrinking the outlines) is ``prepare multiplex
--nuclei-fallback shrink`` and is never chosen here.

The thresholds, and why:

``ACCEPT_ONE_NUCLEUS`` 0.90  nuclei are USED only above this. An outline is its nucleus
    grown outwards, so the true nuclei put exactly one centroid in (nearly) every outline;
    a mask of another slide, or of the same slide in another frame, puts centroids at
    random, which in tissue gives exactly one in well under half of the outlines.
``EXACT_ONE_NUCLEUS`` 0.99, ``EXACT_COUNT_TOL`` 0.01  ``exact`` needs 99 % of outlines with
    exactly one nucleus and the two counts within 1 %: what is left for concave nuclei
    (centroid outside the outline) and the 1 px simplification of the traced polygons.
    Stated before any cohort run; not tuned on results.
``IDENTITY_*``  the nuclei must also BE nuclei, not the cell mask grown from them and not
    a mask in a shifted frame; one centroid per outline cannot tell (the cell mask itself,
    or the nuclei moved by a few pixels, pass it). Over the nuclei that sit alone in an
    outline (:func:`identity`): the median area ratio nucleus / outline at most
    ``IDENTITY_AREA_RATIO_MAX`` 0.90 (an outline is its nucleus grown by up to 10 px: a
    nucleus of radius 12 px gives 0.30, and only a nucleus hemmed in on all sides
    approaches 1; the cell mask gives 1.00); at least ``IDENTITY_INSIDE_MIN`` 0.90 of the
    nuclei's vertices inside their own outline (a nucleus is contained in the outline grown
    from it; the outlines are simplified to 1 px, hence not 1.00); and the median offset of
    the nucleus's centroid from its outline's at most ``IDENTITY_OFFSET_MAX_PX`` 1.0 px on
    each axis (growing is isotropic, so the two centroids coincide but for neighbours in
    the way). Fixed before any cohort run.
``RERUN_MATCH_PX`` 1.0, ``EXACT_MATCH`` 0.99  the run's own cell mask must put a centroid
    within 1 px of 99 % of the outlines' centroids (the polygons are simplified to 1 px,
    which moves a centroid by a fraction of a pixel).

Nothing here was run on the cohort: StarDist is not installed where this was written and
the site's model is not in the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import cells, manifest, outlines, pairs, site_config
from .cases import file_fingerprint, polys, polyset_arrays, write_json, write_npz

REPO = Path(__file__).resolve().parents[2]
SEGMENT_PY = REPO / "bin" / "segment.py"
ENV_MODEL_DIR = "REGBENCH_STARDIST_MODEL_DIR"
ENV_MODEL = "REGBENCH_STARDIST_MODEL"
ENV_PROB_THRESH = "REGBENCH_STARDIST_PROB_THRESH"
ENV_CPU = "REGBENCH_STARDIST_CPU"
# What the pipeline renders for `--seg_method stardist` at the repository's defaults,
# and where each value is from. The arms set none of them (benchmarks/configs/arms.yaml,
# benchmark.config). The site overlay may: `site_config.resolve` reads the model and any
# override of these from it (conf/ieo.config, or REGBENCH_SITE_CONFIG), and `command`
# renders the resolved values, so the re-run matches what the outlines were made with.
STARDIST = {
    "n_tiles": (16, 16),  # nextflow.config: seg_n_tiles_y, seg_n_tiles_x
    "expand_distance": cells.EXPAND_PX,  # nextflow.config: seg_expand_distance
    "pmin": 1.0,  # nextflow.config: seg_pmin
    "pmax": 99.8,  # nextflow.config: seg_pmax
}
PIPELINE_IMAGE = "bolt3x/mirage-stardist:1.0.0"  # lib/SegBackends.groovy
ACCEPT_ONE_NUCLEUS = 0.90
EXACT_ONE_NUCLEUS = 0.99
EXACT_COUNT_TOL = 0.01
RERUN_MATCH_PX = 1.0
IDENTITY_AREA_RATIO_MAX = 0.90
IDENTITY_INSIDE_MIN = 0.90
IDENTITY_OFFSET_MAX_PX = 1.0
EXACT_MATCH = 0.99
NEEDS_RERUN = (
    "the StarDist run of this slide has not run yet "
    "(python -m benchmarks.regbench nuclei, slurm/nuclei.sbatch)"
)
TRACED = ".traced.json"  # beside an outlines GeoJSON this module traced
NOT_RESOLVED = "not looked at yet (python -m benchmarks.regbench nuclei)"
SOURCE_COLUMNS = {"ref": ("reference", "ref"), "mov": ("moving", "mov")}


# ── the slides of a manifest ──────────────────────────────────────────────────
def slides(rows):
    """Every native slide of a manifest once, in a stable order: the unit of this step.

    A slide is ``{"key", "group", "slide", "image", "channel", "outlines"}``; ``key`` is
    ``<group>/<slide>`` and names its folder; ``outlines`` is where its outlines are
    traced (the manifest's ``ref_cells`` / ``mov_cells``).
    """
    found = {}
    for row in rows:
        group = row.get("group") or row["case_id"]
        for image_col, side in SOURCE_COLUMNS.values():
            image = row[image_col]
            name = manifest.slide_name(image)
            one = {
                "key": f"{group}/{name}",
                "group": group,
                "slide": name,
                "image": image,
                "channel": int(row.get(f"{side}_channel") or 0),
                "outlines": row.get(f"{side}_cells") or "",
            }
            seen = found.setdefault(one["key"], one)
            if seen["image"] != image:
                raise SystemExit(
                    f"two slides of {group} are both named {name}: {seen['image']} "
                    f"and {image}. A slide's nuclei are kept under <group>/<slide>."
                )
    return [found[k] for k in sorted(found)]


def store_of(out, slide):
    return Path(out) / slide["group"] / slide["slide"]


def load(store):
    """The record of one slide, or None when there is none."""
    path = Path(store) / "nuclei.json"
    return json.loads(path.read_text()) if path.is_file() else None


def read_polys(store):
    """The usable nuclei of one slide as a ``PolySet``."""
    with np.load(Path(store) / "nuclei.npz") as z:
        return polys({k: z[k] for k in z.files}, "ref")


# ── agreement with the outlines ───────────────────────────────────────────────
def one_nucleus_fraction(outlines, centroids):
    """The share of outlines that hold exactly one of ``centroids``.

    A centroid is tried against the four outlines whose own centroid is nearest: a
    nucleus inside an outline that is not among them is not that outline's nucleus."""
    from scipy.spatial import cKDTree

    outlines = pairs.as_polyset(outlines)
    n = outlines.n_features
    cent = np.asarray(centroids, dtype=float).reshape(-1, 2)
    cent = cent[np.isfinite(cent).all(axis=1)]
    if not n:
        return float("nan")
    if not len(cent):
        return 0.0
    _, own = pairs.cp.feature_area_centroid(outlines)
    ok = np.isfinite(own).all(axis=1)
    index = np.flatnonzero(ok)
    k = min(4, len(index))
    _, near = cKDTree(own[ok]).query(cent, k=k)
    near = index[np.asarray(near).reshape(len(cent), k)]
    points = np.repeat(np.arange(len(cent)), k)
    feats = near.ravel()
    hit = cells.inside(outlines, feats, cent[points])
    held = np.bincount(feats[hit], minlength=n)
    return float((held == 1).mean())


def holders(outlines, centroids):
    """For each centroid, the index of the outline that holds it (the first of the four
    nearest by centroid), or -1."""
    from scipy.spatial import cKDTree

    outlines = pairs.as_polyset(outlines)
    cent = np.asarray(centroids, dtype=float).reshape(-1, 2)
    out = np.full(len(cent), -1, dtype=np.int64)
    good = np.isfinite(cent).all(axis=1)
    _, own = pairs.cp.feature_area_centroid(outlines)
    ok = np.isfinite(own).all(axis=1)
    index = np.flatnonzero(ok)
    if not len(index) or not good.any():
        return out
    k = min(4, len(index))
    _, near = cKDTree(own[ok]).query(cent[good], k=k)
    near = index[np.asarray(near).reshape(int(good.sum()), k)]
    rows = np.flatnonzero(good)
    for j in range(k - 1, -1, -1):  # the nearest holder wins
        hit = cells.inside(outlines, near[:, j], cent[good])
        out[rows[hit]] = near[hit, j]
    return out


def identity(outlines, nuclei):
    """Are these the NUCLEI the outlines were grown from (module docstring,
    ``IDENTITY_*``)? ``{"area_ratio_med", "vertices_inside_frac", "centroid_offset_px",
    "n_paired", "is_nuclei", "why"}`` over the nuclei that sit alone in an outline."""
    outlines, nuclei = pairs.as_polyset(outlines), pairs.as_polyset(nuclei)
    area_n, cent_n = pairs.cp.feature_area_centroid(nuclei)
    area_o, cent_o = pairs.cp.feature_area_centroid(outlines)
    holder = holders(outlines, cent_n)
    alone = np.bincount(holder[holder >= 0], minlength=outlines.n_features) == 1
    paired = (holder >= 0) & alone[np.clip(holder, 0, None)]
    paired &= np.isfinite(area_n) & (area_n > 0)
    out = {
        "n_paired": int(paired.sum()),
        "area_ratio_med": float("nan"),
        "vertices_inside_frac": float("nan"),
        "centroid_offset_px": [float("nan"), float("nan")],
        "area_ratio_max": IDENTITY_AREA_RATIO_MAX,
        "vertices_inside_min": IDENTITY_INSIDE_MIN,
        "centroid_offset_max_px": IDENTITY_OFFSET_MAX_PX,
        "is_nuclei": False,
        "why": "no nucleus sits alone in an outline",
    }
    if not paired.any():
        return out
    ratio = area_n[paired] / area_o[holder[paired]]
    offset = np.median(cent_n[paired] - cent_o[holder[paired]], axis=0)
    vertex_feat = np.repeat(nuclei.ring_feat, np.diff(nuclei.ring_off))
    use = paired[vertex_feat]
    inside = cells.inside(outlines, holder[vertex_feat[use]], nuclei.xy[use])
    out.update(
        area_ratio_med=float(np.median(ratio)),
        vertices_inside_frac=float(inside.mean()) if len(inside) else float("nan"),
        centroid_offset_px=[float(offset[0]), float(offset[1])],
    )
    problems = []
    if not out["area_ratio_med"] <= IDENTITY_AREA_RATIO_MAX:
        problems.append(
            f"its objects are as large as the outlines (median area ratio "
            f"{out['area_ratio_med']:.2f}, at most {IDENTITY_AREA_RATIO_MAX} for nuclei "
            "inside outlines grown by up to 10 px): it looks like the cell mask"
        )
    if not out["vertices_inside_frac"] >= IDENTITY_INSIDE_MIN:
        problems.append(
            f"only {out['vertices_inside_frac']:.2f} of its objects' vertices lie "
            f"inside their own outline (at least {IDENTITY_INSIDE_MIN}): not contained "
            "in the outlines"
        )
    if not max(abs(offset[0]), abs(offset[1])) <= IDENTITY_OFFSET_MAX_PX:
        problems.append(
            f"its objects sit ({offset[0]:+.1f}, {offset[1]:+.1f}) px from the centres "
            f"of their outlines (at most {IDENTITY_OFFSET_MAX_PX} px on each axis): a "
            "mask in a shifted frame"
        )
    out["is_nuclei"] = not problems
    out["why"] = "; ".join(problems)
    return out


def _counts_agree(a, b):
    return bool(max(a, b) and abs(a - b) / max(a, b) <= EXACT_COUNT_TOL)


def label_agreement(outlines, nuclei):
    """How a set of nuclei sits in a slide's outlines."""
    area, cent = pairs.cp.feature_area_centroid(pairs.as_polyset(nuclei))
    n_out, n_nuc = pairs.n_cells(outlines), pairs.n_cells(nuclei)
    frac = one_nucleus_fraction(outlines, cent)
    same = identity(outlines, nuclei)
    return {
        "n_outlines": n_out,
        "n_nuclei": n_nuc,
        "one_nucleus_frac": frac,
        "identity": same,
        "exact": bool(
            frac >= EXACT_ONE_NUCLEUS
            and _counts_agree(n_out, n_nuc)
            and same["is_nuclei"]
        ),
        "accept_threshold": ACCEPT_ONE_NUCLEUS,
        "exact_threshold": EXACT_ONE_NUCLEUS,
        "count_tolerance": EXACT_COUNT_TOL,
    }


def rerun_agreement(outlines, regrown):
    """How the cell mask a re-run grew sits on the outlines: the two counts,
    and the share of published centroids with a re-grown one within ``RERUN_MATCH_PX``."""
    from scipy.spatial import cKDTree

    _, pub = pairs.cp.feature_area_centroid(pairs.as_polyset(outlines))
    _, mine = pairs.cp.feature_area_centroid(pairs.as_polyset(regrown))
    pub = pub[np.isfinite(pub).all(axis=1)]
    mine = mine[np.isfinite(mine).all(axis=1)]
    if len(pub) and len(mine):
        d, _ = cKDTree(mine).query(pub, k=1)
        matched = float((d <= RERUN_MATCH_PX).mean())
    else:
        matched = 0.0
    return {
        "n_cells_rerun": int(len(mine)),
        "centroid_match_frac": matched,
        "match_px": RERUN_MATCH_PX,
        "match_threshold": EXACT_MATCH,
        "cells_agree": bool(
            matched >= EXACT_MATCH and _counts_agree(len(pub), len(mine))
        ),
    }


# ── the re-run ────────────────────────────────────────────────────────────────
def expected_masks(slide, dest):
    """``(nuclei mask, cell mask)`` as ``bin/segment.py`` names them: the image's stem."""
    stem = Path(slide["image"]).stem
    dest = Path(dest)
    return dest / f"{stem}_nuclei_mask.tif", dest / f"{stem}_cell_mask.tif"


def command(slide, dest, env=None):
    """The ``segment.py`` call of one slide, in the order the pipeline renders it
    (``modules/local/segment.nf``: the entry point, ``--image``, ``--output-dir``, the
    backend's flag, then ``ext.args`` of ``conf/modules.config``).

    The model is the site's: ``REGBENCH_STARDIST_MODEL`` (``segmentation_model``) and
    ``REGBENCH_STARDIST_MODEL_DIR`` (``segmentation_model_dir``) when set, else the
    site config's two keys (:func:`settings`). The segmentation settings are the
    repository's defaults unless the site config overrides one, and then the site's
    value is rendered. The pipeline passes ``--dapi-channel 0`` because its
    preprocessing puts the nuclear channel first (and refuses a slide whose channel 0
    is not nuclear); this passes the manifest's index of the nuclear channel, which is
    0 on every slide the pipeline would segment."""
    env = os.environ if env is None else env
    got = settings(env)["values"]
    model = got["segmentation_model"]["value"]
    if not model:
        raise SystemExit(
            f"{ENV_MODEL} is not set: the name of the site's StarDist model "
            f"(the pipeline's segmentation_model), with {ENV_MODEL_DIR} for its "
            "folder (segmentation_model_dir). Neither is in a site config either "
            f"({site_config.ENV_SITE_CONFIG}, else conf/ieo.config of this checkout)."
        )
    model_dir = got["segmentation_model_dir"]["value"]
    prob = got["seg_prob_thresh"]["value"]
    prob = "" if prob is None else str(prob)
    cpu = (env.get(ENV_CPU) or "").strip() not in ("", "0")

    def of(key):
        return str(got[key]["value"])

    return [
        "python",
        str(SEGMENT_PY),
        "--image",
        str(slide["image"]),
        "--output-dir",
        str(dest),
        "--dapi-channel",
        str(int(slide["channel"])),
        *(["--model-dir", model_dir] if model_dir else []),
        "--model-name",
        model,
        "--n-tiles",
        of("seg_n_tiles_y"),
        of("seg_n_tiles_x"),
        "--expand-distance",
        of("seg_expand_distance"),
        "--pmin",
        of("seg_pmin"),
        "--pmax",
        of("seg_pmax"),
        *(["--prob-thresh", prob] if prob else []),
        *([] if cpu else ["--use-gpu"]),
    ]


# ── one slide ─────────────────────────────────────────────────────────────────
def _shape(path):
    from .imageio import shape_of

    return tuple(int(v) for v in shape_of(path)[1:])


def _outlines(slide):
    if not slide["outlines"] or not cells.is_geojson(slide["outlines"]):
        return None
    return cells.cells_from_geojson(slide["outlines"], hw=_shape(slide["image"]))


def _trace(path, a):
    return cells.polys_from_label_image(
        path, getattr(a, "diameter", 20.0), getattr(a, "tile", 2048), a.workers
    )


def traced_record(path):
    """What an outlines GeoJSON this module traced was traced from (its
    ``.traced.json``), or None: a GeoJSON without one is someone else's (an arm's
    published outlines, named by a manifest written before 2026-10-16)."""
    side = Path(str(path) + TRACED)
    if not path or not side.is_file():
        return None
    try:
        return json.loads(side.read_text())
    except ValueError:
        return None


def is_ours(path):
    """Are these outlines this module's to (re)write: not there yet, or traced here."""
    return bool(path) and (not Path(path).exists() or traced_record(path) is not None)


def trace_outlines(slide, cell_mask):
    """Trace a run's cell mask to the slide's outlines (``outlines.trace``) and leave
    beside them what they were traced from; returns that record."""
    tol = outlines.tolerance()
    n = outlines.trace(cell_mask, slide["outlines"], tol)
    made = {
        "outlines_from": "stardist_rerun_cell_mask",
        "cell_mask": str(cell_mask),
        "cell_mask_fingerprint": file_fingerprint(cell_mask),
        "image": slide["image"],
        "image_fingerprint": file_fingerprint(slide["image"]),
        "tolerance": tol,
        "n_outlines": n,
        "traced_by": str(outlines.MASK_TO_GEOJSON),
    }
    write_json(Path(str(slide["outlines"]) + TRACED), made)
    return made


def set_outlines_aside(path):
    """Move outlines THIS module traced out of the way (``<name>.stale``), with their
    record: they belong to a run that is set aside. Anyone else's are left alone."""
    if not path or traced_record(path) is None:
        return
    import os

    for p in (Path(path), Path(str(path) + TRACED)):
        if p.exists():
            os.replace(p, p.with_name(p.name + ".stale"))


def check_rerun(slide, store, a):
    """Read a finished StarDist run of one slide: trace its cell mask to the slide's
    outlines (unless the manifest names someone else's), and check the nuclei against
    them."""
    dest = Path(store) / "rerun"
    nuc_mask, cell_mask = expected_masks(slide, dest)
    rec = {
        **{k: slide[k] for k in ("key", "group", "slide", "image", "channel")},
        "image_fingerprint": file_fingerprint(slide["image"]),
        "label_image": str(nuc_mask),
        "origin": "stardist_rerun",
        "outlines": slide["outlines"],
        "usable": False,
        "exact": False,
        "source": "",
        "needs_rerun": True,
    }
    if not nuc_mask.is_file():
        rec["reason"] = NEEDS_RERUN
        return rec
    want = _shape(slide["image"])
    if _shape(nuc_mask) != want:
        rec["reason"] = f"{nuc_mask} is not of its slide's size ({want[1]} x {want[0]})"
        return rec
    nuclei = _trace(nuc_mask, a)
    if not len(nuclei):
        rec["reason"] = f"{nuc_mask}: StarDist found no nucleus"
        rec["needs_rerun"] = False
        return rec
    own = is_ours(slide["outlines"])
    if own:
        if not cell_mask.is_file():
            rec["reason"] = (
                f"the StarDist run wrote no cell mask ({cell_mask}): the slide's "
                "outlines cannot be traced. Run it again (REGBENCH_NUCLEI_FORCE=1)"
            )
            return rec
        made = trace_outlines(slide, cell_mask)
        rec.update(outlines_from=made["outlines_from"], tolerance=made["tolerance"])
    elif slide["outlines"]:
        rec["outlines_from"] = "manifest"
    outlines_ = _outlines(slide)
    exact, agreement = False, None
    if outlines_ is not None:
        agreement = label_agreement(outlines_, nuclei)
        if cell_mask.is_file():
            agreement.update(rerun_agreement(outlines_, _trace(cell_mask, a)))
        exact = bool(agreement["exact"] and agreement.get("cells_agree"))
        agreement["exact"] = exact
        what = (
            "the outlines traced from its own cell mask"
            if own
            else "the outlines the manifest names"
        )
        if not agreement["one_nucleus_frac"] >= ACCEPT_ONE_NUCLEUS:
            # user ruling 2026-10-11: nuclei this far from the outlines are another
            # segmentation (or, against their own trace, a tracing bug). REFUSED: the
            # slide has no nuclei, and says why wherever nuclei are missing.
            rec.update(
                refused=True,
                needs_rerun=False,
                agreement=agreement,
                **_made_with(slide, dest),
                reason=(
                    f"the StarDist re-run was REFUSED: only "
                    f"{agreement['one_nucleus_frac']:.3f} of this slide's "
                    f"{agreement['n_outlines']} outlines ({what}) hold exactly one "
                    f"of its {agreement['n_nuclei']} nuclei (at least "
                    f"{ACCEPT_ONE_NUCLEUS} is required)"
                    + (
                        ": the trace does not reproduce the mask it was traced "
                        "from, a tracing bug"
                        if own
                        else ": not the segmentation the outlines were made with. "
                        "Check the model, the channel and the site's segmentation "
                        "settings"
                    )
                ),
            )
            (Path(store) / "nuclei.npz").unlink(missing_ok=True)
            print(f"REFUSED {slide['key']}: {rec['reason']}", flush=True)
            return rec
        if not exact:
            print(
                f"DISAGREEMENT {slide['key']}: the StarDist run does not reproduce "
                f"{what} ({json.dumps(agreement)}). Its nuclei are kept and marked "
                "exact: false. "
                + (
                    "The outlines are its own trace, so this is a tracing bug: check "
                    "outlines.trace before reading nuclei numbers."
                    if own
                    else "Check the model, the channel and the pipeline's "
                    "segmentation settings before reading nuclei numbers."
                ),
                flush=True,
            )
    rec.update(
        usable=True,
        needs_rerun=False,
        exact=exact,
        agreement=agreement,
        source="stardist_rerun",
        **_made_with(slide, dest),
    )
    _keep(store, rec, nuclei)
    return rec


def settings(env=None):
    """The model and settings of the re-run with the source of each
    (``site_config.resolve``): the environment, a site config, the repository."""
    return site_config.resolve(env)


def _settings_or_none():
    try:
        return settings()
    except SystemExit as exc:
        return {"error": str(exc)}


def _command_or_none(slide, dest):
    try:
        return command(slide, dest)
    except SystemExit:
        return None


PRODUCED = "produced_by.json"
AT_SEGMENTATION = "when the re-run was rendered (rerun/produced_by.json)"
AT_CHECK = "when the mask was checked: no record of the segmentation itself was found"


def write_produced(slide, dest):
    """Leave, beside the masks a re-run is about to write, the command and the settings
    it is rendered with NOW. A record made later (when the mask is checked) could carry
    the settings of another day."""
    record = {"command": shlex.join(command(slide, dest)), "settings": settings()}
    write_json(Path(dest) / PRODUCED, record)
    return record


def _made_with(slide, dest):
    """``settings`` / ``command`` / ``settings_recorded`` of a re-run's record: what the
    masks under ``dest`` were PRODUCED with, when that was written down."""
    path = Path(dest) / PRODUCED
    if path.is_file():
        try:
            made = json.loads(path.read_text())
            return {
                "settings": made["settings"],
                "command": made["command"],
                "settings_recorded": AT_SEGMENTATION,
            }
        except (ValueError, KeyError):
            pass
    return {
        "settings": _settings_or_none(),
        "command": shlex.join(_command_or_none(slide, dest) or []),
        "settings_recorded": AT_CHECK,
    }


def _values(resolved):
    return {
        k: str(v.get("value", ""))
        for k, v in ((resolved or {}).get("values") or {}).items()
    }


def settings_changed(rec):
    """What differs between the settings a kept re-run was made with and the ones that
    resolve now, as ``["key: old -> new"]``. Empty for a record that is no re-run, that
    recorded no settings, or when nothing resolves now (no model named)."""
    if not rec or not (rec.get("source") == "stardist_rerun" or rec.get("refused")):
        return []
    old = _values(rec.get("settings"))
    now = _values(_settings_or_none())
    if not old or not now:
        return []
    return [
        f"{k}: {old.get(k, '')!r} -> {now.get(k, '')!r}"
        for k in sorted(set(old) | set(now))
        if old.get(k, "") != now.get(k, "")
    ]


def set_aside(store, traced=""):
    """Move a re-run that is no longer the one to use out of the way (``rerun_stale/``,
    the one before it removed) and drop its polygons, and the outlines ``traced`` from
    it when this module traced them (:func:`set_outlines_aside`). The record goes
    FIRST, in one rename (``nuclei.stale.json``): a task that is ended anywhere after
    that leaves a slide without a record, which is decided again, never a record that
    says "usable" over polygons that are gone."""
    import os
    import shutil

    store = Path(store)
    stale = store / "rerun_stale"
    if (store / "nuclei.json").is_file():
        os.replace(store / "nuclei.json", store / "nuclei.stale.json")
    if (store / "rerun").is_dir():
        shutil.rmtree(stale, ignore_errors=True)
        (store / "rerun").rename(stale)
    (store / "nuclei.npz").unlink(missing_ok=True)
    set_outlines_aside(traced)


def outlines_moved(rec, slide):
    """A usable record whose outlines are not there any more, or not where the manifest
    names them now: the kept run is read again (and traced again where it is ours)."""
    path = slide["outlines"]
    if not path:
        return False
    return not Path(path).is_file() or rec.get("outlines", path) != path


def _keep(store, rec, nuclei):
    store = Path(store)
    write_npz(store / "nuclei.npz", **polyset_arrays(nuclei, "ref"))
    rec["n_nuclei"] = int(pairs.n_cells(nuclei))


def other_image(rec, slide):
    """Why a record is not of the files that are there now, or ``""``: the slide (or
    the label image the nuclei were traced from) has another size, time or sampled
    bytes than when the record was made, or the record does not say what it was made
    from. Such a record is never used."""
    if not rec:
        return ""
    have = rec.get("image_fingerprint")
    if not have:
        return "its record does not say which image it was computed from"
    if have != file_fingerprint(slide["image"]):
        return (
            f"the slide {slide['image']} is not the file its nuclei were computed from"
        )
    if str(rec.get("source") or "").startswith("label_image:"):
        path = rec.get("label_image") or ""
        if rec.get("label_image_fingerprint") != file_fingerprint(path):
            return f"the label image {path} is not the file its nuclei were traced from"
    return ""


def resolve(slide, out, a):
    """Decide one slide and write its record: a finished StarDist run (its outlines
    traced, its nuclei checked), else ``needs_rerun``.

    What is kept is used only when it is of the files that are there now
    (:func:`other_image`), its run was made with the settings that resolve now, and the
    outlines it was checked against are still where the manifest names them.
    ``a.force`` decides the slide again whatever is kept. In the first two cases and
    under ``force`` nothing kept answers: the run is set aside (``rerun_stale/``, the
    outlines traced from it with it), so the slide needs its StarDist run again. When
    only the outlines are missing (or the manifest names another place for them), the
    kept run is read again and traced there (no GPU).

    A record that does not say which image it was computed from is neither used as it
    is nor thrown away: the run's masks kept for it are checked again against the slide
    that is there now (no GPU), and a record that passes carries the fingerprint from
    then on. Only what fails that check is set aside."""
    store = store_of(out, slide)
    old = load(store)
    force = getattr(a, "force", False)
    # A re-run kept here (usable, or refused) was made with the model and settings of
    # ITS day. When they resolve to something else now, it is not the re-run to keep:
    # said aloud, set aside, and the slide needs its re-run again.
    changed = [] if force else settings_changed(old)
    moved = "" if force else other_image(old, slide)
    unstamped = bool(moved and old and not old.get("image_fingerprint") and not changed)
    if unstamped:
        moved = ""
        print(
            f"RECHECK {slide['key']}: its record does not say which image it was "
            "computed from. What is kept for it is checked again against the slide "
            "(StarDist is not run for that).",
            flush=True,
        )
    if changed:
        print(
            f"STALE {slide['key']}: its StarDist re-run was made with other settings "
            f"({'; '.join(changed)}). It is set aside (rerun_stale/) and the slide is "
            "segmented again.",
            flush=True,
        )
    elif moved:
        print(
            f"STALE {slide['key']}: {moved}. What is kept for it is set aside "
            "(rerun_stale/) and the slide is decided again.",
            flush=True,
        )
    elif force and old:
        print(
            f"FORCED {slide['key']}: what is kept for it is set aside (rerun_stale/) "
            "and the slide is decided again.",
            flush=True,
        )
    if changed or moved or force:
        set_aside(store, slide["outlines"])
        old = None
    if unstamped:
        old = None
    if old and old.get("usable") and not outlines_moved(old, slide):
        return old
    rec = check_rerun(slide, store, a)
    if changed and rec.get("needs_rerun"):
        rec["stale_settings"] = changed
        rec["reason"] = (
            "the re-run kept for this slide was made with other settings "
            f"({'; '.join(changed)}); it is set aside and must be run again"
        )
    elif moved and rec.get("needs_rerun"):
        rec["stale_image"] = moved
        rec["reason"] = f"{moved}: what was kept for it is set aside; {rec['reason']}"
    elif unstamped and rec.get("needs_rerun") and (Path(store) / "rerun").is_dir():
        # checked again, and it is not this slide's: a real mismatch
        set_aside(store, slide["outlines"])
        rec["reason"] = (
            "the record kept for this slide did not say which image it was computed "
            f"from, and what was kept does not fit the slide: set aside; {rec['reason']}"
        )
    write_json(store / "nuclei.json", rec)
    return rec


def segment(slide, out):
    """Run the re-run of one slide here (the image that has StarDist); returns its exit
    code. On a cluster ``slurm/nuclei.sbatch`` runs the printed command instead."""
    dest = store_of(out, slide) / "rerun"
    dest.mkdir(parents=True, exist_ok=True)
    argv = command(slide, dest)
    write_produced(slide, dest)
    argv[0] = sys.executable
    print("+ " + shlex.join(argv), flush=True)
    return subprocess.run(argv, check=False).returncode


# ── the table ─────────────────────────────────────────────────────────────────
COLUMNS = (
    "index",
    "slide",
    "status",
    "source",
    "exact",
    "one_nucleus_frac",
    "n_nuclei",
    "reason",
)


def table(all_slides, out):
    """One row per slide: what its nuclei are, or why it has none yet."""
    rows = []
    for i, slide in enumerate(all_slides):
        rec = load(store_of(out, slide))
        if rec is None:
            status, rec = "needs_rerun", {"reason": NOT_RESOLVED}
        elif rec.get("usable"):
            status = "usable"
        elif rec.get("refused"):
            status = "refused"
        elif rec.get("needs_rerun"):
            status = "needs_rerun"
        else:
            status = "none"
        frac = (rec.get("agreement") or {}).get("one_nucleus_frac")
        rows.append(
            {
                "index": i,
                "slide": slide["key"],
                "status": status,
                "source": rec.get("source", ""),
                "exact": rec.get("exact", ""),
                "one_nucleus_frac": "" if frac is None else f"{frac:.4f}",
                "n_nuclei": rec.get("n_nuclei", ""),
                "reason": rec.get("reason", ""),
            }
        )
    return rows


def main(a):
    """``python -m benchmarks.regbench nuclei`` (``cli.py`` has the options)."""
    from .datasets.multiplex import read_manifest

    if getattr(a, "settings", False):
        # the model and settings of the re-run and where each is from (the launcher's
        # dry run prints this); a model nothing names is an error here, before any job
        got = settings()
        for line in site_config.lines(got):
            print(line)
        if not got["values"]["segmentation_model"]["value"]:
            raise SystemExit(
                f"no StarDist model: {ENV_MODEL} is not set and no site config names "
                f"segmentation_model ({site_config.ENV_SITE_CONFIG}, else "
                "conf/ieo.config of this checkout)."
            )
        return 0
    all_slides = slides(read_manifest(a.manifest))
    if a.count:
        print(len(all_slides))
        return 0
    if a.list:
        rows = table(all_slides, a.out)
        print("\t".join(COLUMNS))
        for r in rows:
            print("\t".join(str(r[c]) for c in COLUMNS))
        kinds = ("usable", "needs_rerun", "refused")
        n = {s: sum(r["status"] == s for r in rows) for s in kinds}
        print(
            f"# {len(rows)} slide(s): {n['usable']} with usable nuclei, "
            f"{n['needs_rerun']} needing the StarDist re-run, "
            f"{n['refused']} whose re-run was REFUSED, "
            f"{len(rows) - sum(n.values())} other",
            file=sys.stderr,
        )
        return 0
    chosen = all_slides
    if a.index is not None:
        if not 0 <= a.index < len(all_slides):
            raise SystemExit(f"--index {a.index}: the manifest has {len(all_slides)}")
        chosen = [all_slides[a.index]]
    if a.command:
        # what a slide still needs, as one shell line each; nothing for a slide that
        # has usable nuclei (the sbatch script runs the line in the StarDist image)
        said = False
        for slide in chosen:
            rec = load(store_of(a.out, slide))
            # nothing either for a slide whose re-run was refused: running the same
            # command again gives the same nuclei. A slide decided with --force has
            # had its re-run set aside (resolve), so it is not usable here and gets
            # its line
            if rec and (rec.get("usable") or rec.get("refused")):
                continue
            dest = store_of(a.out, slide) / "rerun"
            dest.mkdir(parents=True, exist_ok=True)
            # the line is about to be run: what it is rendered with is written down
            line = write_produced(slide, dest)["command"]
            if not said:  # on stderr: stdout is the command, one shell line a slide
                for text in site_config.lines(settings()):
                    print("# " + text, file=sys.stderr)
                said = True
            print(line)
        return 0
    bad = 0
    # the masks a re-run has just written are read as they are: only the FIRST look at
    # a slide is forced
    after = argparse.Namespace(**{**vars(a), "force": False})
    for slide in chosen:
        rec = resolve(slide, a.out, a)
        if not rec["usable"] and a.segment and rec.get("needs_rerun"):
            code = segment(slide, a.out)
            if code:
                print(f"nuclei/{slide['key']}: segment.py exited {code}")
            rec = resolve(slide, a.out, after)
        state = rec["source"] if rec["usable"] else f"NONE ({rec.get('reason', '')})"
        print(f"nuclei/{slide['key']}: {state} exact={rec['exact']}", flush=True)
        bad += not rec["usable"] and not rec.get("needs_rerun")
    return 1 if chosen and bad == len(chosen) else 0
