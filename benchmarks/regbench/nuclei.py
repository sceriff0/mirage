"""Bare nuclei for the cohort's slides: reuse what exists, segment only what has none.

The cohort's published outlines are StarDist nuclei grown by 10 px (README, "What the
cohort's nuclei are"); the nuclei themselves are not among an arm's published files. This
module finds or makes, for each NATIVE slide of a ``prepare multiplex`` manifest, a label
image of its bare nuclei, checks it against that slide's published outlines, and leaves
one record per slide that ``prepare multiplex --nuclei-dir`` reads::

    <out>/<group>/<slide>/nuclei.json   where the nuclei came from, whether they may be
                                        used, how well they agree with the outlines
    <out>/<group>/<slide>/nuclei.npz    the nuclei as polygons (when usable)
    <out>/<group>/<slide>/rerun/        the masks of a StarDist re-run of that slide

Sources, tried in this order for each slide:

1. **an existing label image** named by the manifest (``ref_nuclei`` / ``mov_nuclei``;
   ``arms_manifest --nuclei-glob`` finds masks kept in a Nextflow work directory,
   ``--seg-arm`` takes a segmentation arm's per-patient mask for the reference slide). It
   is refused unless its shape is the slide's and ``ACCEPT_ONE_NUCLEUS`` of the slide's
   outlines hold exactly one of its nuclei's centroids (:func:`one_nucleus_fraction`);
2. **a StarDist re-run**: the pipeline's ``bin/segment.py``, unchanged, on the native
   slide with the arguments the pipeline renders for the arms (:func:`command`). It
   writes the nuclei mask and the cell mask it grows from it; the cell mask is compared
   with the published outlines (:func:`rerun_agreement`).

Anything else (shrinking the outlines) is ``prepare multiplex --nuclei-fallback shrink``
and is never chosen here. ``exact`` in a record means: these nuclei are, within the stated
tolerances, the ones the published outlines were grown from.

The thresholds, and why:

``ACCEPT_ONE_NUCLEUS`` 0.90  a label image is USED only above this. An outline is its
    nucleus grown outwards, so the true nuclei put exactly one centroid in (nearly) every
    outline; a mask of another slide, or of the same slide in another frame, puts centroids
    at random, which in tissue gives exactly one in well under half of the outlines.
``EXACT_ONE_NUCLEUS`` 0.99, ``EXACT_COUNT_TOL`` 0.01  ``exact`` needs 99 % of outlines with
    exactly one nucleus and the two counts within 1 %: what is left for concave nuclei
    (centroid outside the outline), the 1 px simplification of the published polygons and
    a GPU's run-to-run differences. Stated before any cohort run; not tuned on results.
``RERUN_MATCH_PX`` 1.0, ``EXACT_MATCH`` 0.99  a re-run's own cell mask must put a
    centroid within 1 px of 99 % of the published outlines' centroids (the published
    polygons are simplified to 1 px, which moves a centroid by a fraction of a pixel).

Nothing here was run on the cohort: StarDist is not installed where this was written, the
site's model is not in the repository, and no arm's files were read.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import arms_manifest, cells, pairs
from .cases import polys, polyset_arrays, write_json, write_npz

REPO = Path(__file__).resolve().parents[2]
SEGMENT_PY = REPO / "bin" / "segment.py"
ENV_MODEL_DIR = "REGBENCH_STARDIST_MODEL_DIR"
ENV_MODEL = "REGBENCH_STARDIST_MODEL"
ENV_PROB_THRESH = "REGBENCH_STARDIST_PROB_THRESH"
ENV_CPU = "REGBENCH_STARDIST_CPU"
# What the pipeline renders for `--seg_method stardist`, and where each value is from.
# The arms set none of them (benchmarks/configs/arms.yaml, benchmark.config); the site
# overlay (conf/ieo.config) is not in the repository, so a value it overrides would not
# show here: the agreement check of every re-run is what would catch it.
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
EXACT_MATCH = 0.99
NEEDS_RERUN = (
    "no usable nuclear label image for this slide; the StarDist re-run has not run "
    "(python -m benchmarks.regbench nuclei, slurm/nuclei.sbatch)"
)
NOT_RESOLVED = "not looked at yet (python -m benchmarks.regbench nuclei)"
SOURCE_COLUMNS = {"ref": ("reference", "ref"), "mov": ("moving", "mov")}


# ── the slides of a manifest ──────────────────────────────────────────────────
def slides(rows):
    """Every native slide of a manifest once, in a stable order: the unit of this step.

    A slide is ``{"key", "group", "slide", "image", "channel", "outlines", "candidate",
    "candidate_from"}``; ``key`` is ``<group>/<slide>`` and names its folder.
    """
    found = {}
    for row in rows:
        group = row.get("group") or row["case_id"]
        for image_col, side in SOURCE_COLUMNS.values():
            image = row[image_col]
            name = arms_manifest.slide_name(image)
            one = {
                "key": f"{group}/{name}",
                "group": group,
                "slide": name,
                "image": image,
                "channel": int(row.get(f"{side}_channel") or 0),
                "outlines": row.get(f"{side}_cells") or "",
                "candidate": row.get(f"{side}_nuclei") or "",
                "candidate_from": row.get(f"{side}_nuclei_from") or "",
            }
            seen = found.setdefault(one["key"], one)
            if seen["image"] != image:
                raise SystemExit(
                    f"two slides of {group} are both named {name}: {seen['image']} "
                    f"and {image}. A slide's nuclei are kept under <group>/<slide>."
                )
            if not seen["candidate"] and one["candidate"]:
                seen.update(
                    candidate=one["candidate"], candidate_from=one["candidate_from"]
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


# ── agreement with the published outlines ─────────────────────────────────────
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


def _counts_agree(a, b):
    return bool(max(a, b) and abs(a - b) / max(a, b) <= EXACT_COUNT_TOL)


def label_agreement(outlines, nuclei):
    """How a set of nuclei sits in a slide's published outlines."""
    area, cent = pairs.cp.feature_area_centroid(pairs.as_polyset(nuclei))
    n_out, n_nuc = pairs.n_cells(outlines), pairs.n_cells(nuclei)
    frac = one_nucleus_fraction(outlines, cent)
    return {
        "n_outlines": n_out,
        "n_nuclei": n_nuc,
        "one_nucleus_frac": frac,
        "exact": bool(frac >= EXACT_ONE_NUCLEUS and _counts_agree(n_out, n_nuc)),
        "accept_threshold": ACCEPT_ONE_NUCLEUS,
        "exact_threshold": EXACT_ONE_NUCLEUS,
        "count_tolerance": EXACT_COUNT_TOL,
    }


def rerun_agreement(outlines, regrown):
    """How the cell mask a re-run grew sits on the published outlines: the two counts,
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

    The model is the site's and comes from the environment: ``REGBENCH_STARDIST_MODEL``
    (``segmentation_model``) and ``REGBENCH_STARDIST_MODEL_DIR``
    (``segmentation_model_dir``). The pipeline passes ``--dapi-channel 0`` because its
    preprocessing puts the nuclear channel first; this passes the manifest's index of
    the nuclear channel, which is 0 on the arms' slides."""
    env = os.environ if env is None else env
    model = (env.get(ENV_MODEL) or "").strip()
    if not model:
        raise SystemExit(
            f"{ENV_MODEL} is not set: the name of the StarDist model the arms ran "
            f"(the pipeline's segmentation_model), with {ENV_MODEL_DIR} for its "
            "folder (segmentation_model_dir)."
        )
    model_dir = (env.get(ENV_MODEL_DIR) or "").strip()
    prob = (env.get(ENV_PROB_THRESH) or "").strip()
    cpu = (env.get(ENV_CPU) or "").strip() not in ("", "0")
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
        *(str(v) for v in STARDIST["n_tiles"]),
        "--expand-distance",
        str(STARDIST["expand_distance"]),
        "--pmin",
        str(STARDIST["pmin"]),
        "--pmax",
        str(STARDIST["pmax"]),
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


def check_label_image(slide, path, store, a, outlines=None, origin=""):
    """Try one existing label image as a slide's nuclei; the record says whether it may
    be used and why not. ``origin`` is how the manifest came by it."""
    rec = {
        **{k: slide[k] for k in ("key", "group", "slide", "image", "channel")},
        "label_image": str(path),
        "origin": origin,
        "usable": False,
        "exact": False,
        "source": "",
        "needs_rerun": True,
    }
    path = Path(path)
    if not path.is_file():
        rec["reason"] = f"the label image is not on disk: {path}"
        return rec
    try:
        shape, want = _shape(path), _shape(slide["image"])
    except Exception as exc:  # an unreadable file is a refusal, not a crash
        rec["reason"] = f"{path}: cannot be read ({type(exc).__name__}: {exc})"
        return rec
    if shape != want:
        rec["reason"] = (
            f"{path} is {shape[1]} x {shape[0]} px and its slide {want[1]} x "
            f"{want[0]} px: not a mask of this native slide"
        )
        return rec
    nuclei = _trace(path, a)
    if not len(nuclei):
        rec["reason"] = f"{path}: no label"
        return rec
    outlines = _outlines(slide) if outlines is None else outlines
    if outlines is None:
        # nothing to check against: the mask is taken as given, and said to be unchecked
        rec.update(agreement=None, exact=False)
    else:
        got = label_agreement(outlines, nuclei)
        rec["agreement"] = got
        if not got["one_nucleus_frac"] >= ACCEPT_ONE_NUCLEUS:
            rec["reason"] = (
                f"{path}: only {got['one_nucleus_frac']:.3f} of this slide's "
                f"{got['n_outlines']} published outlines hold exactly one of its "
                f"{got['n_nuclei']} nuclei (at least {ACCEPT_ONE_NUCLEUS} is "
                "required): not the nuclei of this slide in its native frame"
            )
            return rec
        rec["exact"] = got["exact"]
    rec.update(usable=True, needs_rerun=False, source=f"label_image:{path}")
    _keep(store, rec, nuclei)
    return rec


def check_rerun(slide, store, a):
    """Read a finished re-run of one slide and compare it with the published outlines."""
    dest = Path(store) / "rerun"
    nuc_mask, cell_mask = expected_masks(slide, dest)
    rec = {
        **{k: slide[k] for k in ("key", "group", "slide", "image", "channel")},
        "label_image": str(nuc_mask),
        "origin": "stardist_rerun",
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
    outlines = _outlines(slide)
    exact, agreement = False, None
    if outlines is not None:
        agreement = label_agreement(outlines, nuclei)
        if cell_mask.is_file():
            agreement.update(rerun_agreement(outlines, _trace(cell_mask, a)))
        exact = bool(agreement["exact"] and agreement.get("cells_agree"))
        agreement["exact"] = exact
        if not exact:
            print(
                f"DISAGREEMENT {slide['key']}: the re-run does not reproduce the "
                f"published outlines ({json.dumps(agreement)}). Its nuclei are kept "
                "and marked exact: false. Check the model, the channel and the "
                "pipeline's segmentation settings before reading nuclei numbers.",
                flush=True,
            )
    rec.update(
        usable=True,
        needs_rerun=False,
        exact=exact,
        agreement=agreement,
        source="stardist_rerun",
        command=shlex.join(_command_or_none(slide, dest) or []),
    )
    _keep(store, rec, nuclei)
    return rec


def _command_or_none(slide, dest):
    try:
        return command(slide, dest)
    except SystemExit:
        return None


def _keep(store, rec, nuclei):
    store = Path(store)
    write_npz(store / "nuclei.npz", **polyset_arrays(nuclei, "ref"))
    rec["n_nuclei"] = int(pairs.n_cells(nuclei))


def resolve(slide, out, a):
    """Decide one slide and write its record: a usable existing label image, else a
    finished re-run, else ``needs_rerun``."""
    store = store_of(out, slide)
    old = load(store)
    if old and old.get("usable") and not getattr(a, "force", False):
        return old
    rec = None
    if slide["candidate"]:
        rec = check_label_image(
            slide, slide["candidate"], store, a, origin=slide["candidate_from"]
        )
    if rec is None or not rec["usable"]:
        refused = rec["reason"] if rec else ""
        rec = check_rerun(slide, store, a)
        if refused:
            rec["candidate_refused"] = refused
            if not rec["usable"]:
                rec["reason"] = f"{refused}; {rec['reason']}"
    write_json(store / "nuclei.json", rec)
    return rec


def segment(slide, out):
    """Run the re-run of one slide here (the image that has StarDist); returns its exit
    code. On a cluster ``slurm/nuclei.sbatch`` runs the printed command instead."""
    dest = store_of(out, slide) / "rerun"
    dest.mkdir(parents=True, exist_ok=True)
    argv = command(slide, dest)
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
            status, rec = (
                ("candidate", {"reason": NOT_RESOLVED})
                if slide["candidate"]
                else ("needs_rerun", {"reason": NOT_RESOLVED})
            )
        elif rec.get("usable"):
            status = "usable"
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

    all_slides = slides(read_manifest(a.manifest))
    if a.count:
        print(len(all_slides))
        return 0
    if a.list:
        rows = table(all_slides, a.out)
        print("\t".join(COLUMNS))
        for r in rows:
            print("\t".join(str(r[c]) for c in COLUMNS))
        n = {s: sum(r["status"] == s for r in rows) for s in ("usable", "needs_rerun")}
        print(
            f"# {len(rows)} slide(s): {n['usable']} with usable nuclei, "
            f"{n['needs_rerun']} needing the StarDist re-run, "
            f"{len(rows) - n['usable'] - n['needs_rerun']} other",
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
        for slide in chosen:
            rec = load(store_of(a.out, slide))
            if rec and rec.get("usable") and not a.force:
                continue
            dest = store_of(a.out, slide) / "rerun"
            dest.mkdir(parents=True, exist_ok=True)
            print(shlex.join(command(slide, dest)))
        return 0
    bad = 0
    for slide in chosen:
        rec = resolve(slide, a.out, a)
        if not rec["usable"] and a.segment:
            code = segment(slide, a.out)
            if code:
                print(f"nuclei/{slide['key']}: segment.py exited {code}")
            rec = resolve(slide, a.out, a)
        state = rec["source"] if rec["usable"] else f"NONE ({rec.get('reason', '')})"
        print(f"nuclei/{slide['key']}: {state} exact={rec['exact']}", flush=True)
        bad += not rec["usable"] and not rec.get("needs_rerun")
    return 1 if chosen and bad == len(chosen) else 0
