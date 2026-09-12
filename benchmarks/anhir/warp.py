"""Push each case's source landmarks into the target frame through a method's transform.

Four "methods" produce a warped-landmark directory of the same shape
(``<out>/<case_id>.csv`` in the challenge's ``,X,Y`` format, plus
``<out>/warp_index.csv`` recording paths, times and failures):

``tiled``     the pipeline's STARE backend. Its published transform is
              ``<outdir>/<patient>/registered/manifest/*_manifest.json`` -- a global
              affine plus a control-grid mesh -- warped with the same
              ``bin/utils/tiled_stage_warp.make_warper`` the pipeline's reg_qc=2
              scorer uses. Pure NumPy, runs anywhere.
``valis``     the pipeline's VALIS backend. Its transform is
              ``<outdir>/<patient>/registered/transform/*_registrar.pickle``;
              unpickling it needs the ``valis`` package AND a BioFormats JVM, so
              run this method inside the pipeline's VALIS container (see README).
``bunwarpj``  the challenge's own bUnwarpJ baseline, read from the folder ANHIR
              distributes (``<root>/<case_id>/warped_source_landmarks.txt`` in
              ImageJ point format, ``TIME.txt`` in milliseconds).
``initial``   no registration: the source landmarks copied through. This is the
              pose robustness is measured against, and the floor every method
              must beat.

Slide naming differs per backend and is resolved rather than assumed: a STARE
manifest names the moving slide by its channel list, VALIS by the file stem.
With two slides per case there is exactly one non-reference slide, so the
preferred name from the pairs manifest is used when present and the single
remaining slide otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .landmarks import read_imagej_points, read_landmarks, write_landmarks

METHODS = ("initial", "tiled", "valis", "bunwarpj")
PIPELINE_METHODS = ("tiled", "valis")

REPO = Path(__file__).resolve().parents[2]
BIN_UTILS = REPO / "bin" / "utils"

PAIRS_COLUMNS = [
    "case_id",
    "patient_id",
    "status",
    "tissue",
    "scale",
    "source_image",
    "target_image",
    "source_landmarks",
    "target_landmarks",
    "diagonal",
    "moving_stem",
    "reference_stem",
    "moving_channels",
    "reference_channels",
]

INDEX_COLUMNS = ["case_id", "method", "warped_path", "time_min", "error"]

TRANSFORM_GLOB = {
    "tiled": "registered/manifest/*_manifest.json",
    "valis": "registered/transform/*_registrar.pickle",
}

# Registration processes whose realtime adds up to a case's execution time.
REGISTRATION_PROCESSES = ("REGISTER", "TILED_COARSE", "TILED_REG_TILE", "TILED_SOLVE")


def load_pairs(pairs_csv) -> pd.DataFrame:
    df = pd.read_csv(pairs_csv)
    missing = [c for c in PAIRS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{pairs_csv}: missing pairs column(s) {missing}")
    return df


def find_transform(outdir, patient_id: str, method: str) -> Path:
    pattern = TRANSFORM_GLOB[method]
    hits = sorted((Path(outdir) / str(patient_id)).glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"no {method} transform for {patient_id}: nothing matches "
            f"{Path(outdir) / str(patient_id) / pattern}"
        )
    if len(hits) > 1:
        raise RuntimeError(f"{len(hits)} {method} transforms for {patient_id}: {hits}")
    return hits[0]


def pick_moving(names, ref_name, preferred=None) -> str:
    """The moving slide's key: the preferred name if present, else the one non-reference."""
    names = list(names)
    if preferred in names:
        return preferred
    others = [n for n in names if n != ref_name]
    if len(others) == 1:
        return others[0]
    raise KeyError(
        f"cannot pick the moving slide: preferred {preferred!r} absent and "
        f"{len(others)} non-reference slides among {names} (ref {ref_name!r})"
    )


# ── tiled / STARE ─────────────────────────────────────────────────────────────
def load_tiled_warper(manifest_path):
    if str(BIN_UTILS) not in sys.path:
        sys.path.insert(0, str(BIN_UTILS))
    from tiled_stage_warp import make_warper  # noqa: E402  (numpy-only STARE code)

    manifest = json.loads(Path(manifest_path).read_text())
    return manifest, make_warper(manifest)


def warp_tiled(manifest_path, moving_name, xy, stage: str = "refined") -> np.ndarray:
    manifest, warp = load_tiled_warper(manifest_path)
    name = pick_moving(manifest["slides"], manifest.get("ref_slide"), moving_name)
    return np.asarray(warp(name, np.asarray(xy, dtype=float), stage), dtype=float)


# ── valis ─────────────────────────────────────────────────────────────────────
def _start_jvm(pickle_path) -> None:
    """VALIS slides unpickle against a live BioFormats JVM; start it the way the pipeline does."""
    if str(BIN_UTILS) not in sys.path:
        sys.path.insert(0, str(BIN_UTILS))
    try:
        from valis_config import init_jvm  # the pipeline's sizer (read-only $HOME safe)

        init_jvm(str(Path(pickle_path).resolve().parent))
    except ImportError:
        from valis import registration

        registration.init_jvm()


def warp_valis(pickle_path, moving_name, xy, non_rigid: bool = True) -> np.ndarray:
    """Warp through the VALIS registrar, into the REFERENCE slide's own pixel frame.

    ``crop="reference"`` matters: VALIS's registered canvas is padded, so the
    default crop would return coordinates offset by the reference's position in
    that canvas rather than in the target image the landmarks were placed on.
    """
    _start_jvm(pickle_path)
    from valis import registration

    registrar = registration.load_registrar(str(pickle_path))
    ref_name = registrar.get_ref_slide().name
    name = pick_moving(registrar.slide_dict, ref_name, moving_name)
    slide = registrar.slide_dict[name]
    warped = slide.warp_xy(
        np.asarray(xy, dtype=float), non_rigid=non_rigid, crop="reference"
    )
    return np.asarray(warped, dtype=float)


# ── bUnwarpJ baseline ─────────────────────────────────────────────────────────
def bunwarpj_case(baseline_root, case_id) -> tuple:
    """``(warped_xy, time_min)`` from the challenge's baseline folder for one case."""
    d = Path(baseline_root) / str(int(case_id))
    pts = d / "warped_source_landmarks.txt"
    if not pts.exists():
        raise FileNotFoundError(
            f"baseline has no warped landmarks for case {case_id}: {pts}"
        )
    xy = read_imagej_points(pts)
    t = d / "TIME.txt"
    time_min = float("nan")
    if t.exists():
        raw = t.read_text().strip()
        if raw:
            time_min = float(raw) / 1000.0 / 60.0  # ImageJ getTime() is milliseconds
    return xy, time_min


# ── execution time from the pipeline trace ────────────────────────────────────
def times_from_trace(trace_txt, patient_ids) -> dict:
    """Per-patient minutes spent in registration processes, from ``-with-trace``.

    Tags are matched by containment of the patient id, which is how every
    registration process tags itself; a trace without tags yields no times.
    """
    from benchmarks.analysis.lib.load import parse_trace

    df = parse_trace(trace_txt)
    if "tag" not in df or df["tag"].isna().all():
        return {}
    proc = df["process"].astype(str).str.split(":").str[-1]
    reg = df[proc.isin(REGISTRATION_PROCESSES)]
    out = {}
    for pid in patient_ids:
        rows = reg[reg["tag"].astype(str).str.contains(str(pid), regex=False)]
        if len(rows):
            out[str(pid)] = float(rows["realtime_s"].sum() / 60.0)
    return out


# ── driver ────────────────────────────────────────────────────────────────────
def warp_all(
    method: str,
    pairs: pd.DataFrame,
    out_dir,
    outdir=None,
    baseline_root=None,
    trace=None,
    stage: str = "refined",
) -> pd.DataFrame:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = {}
    if method in PIPELINE_METHODS:
        if outdir is None:
            raise ValueError(f"--outdir is required for method {method}")
        if trace:
            times = times_from_trace(trace, pairs["patient_id"].astype(str))
    if method == "bunwarpj" and baseline_root is None:
        raise ValueError("--baseline-root is required for method bunwarpj")

    rows = []
    for rec in pairs.itertuples(index=False):
        case_id = int(rec.case_id)
        row = {
            "case_id": case_id,
            "method": method,
            "warped_path": "",
            "time_min": float("nan"),
            "error": "",
        }
        try:
            if method == "bunwarpj":
                xy, row["time_min"] = bunwarpj_case(baseline_root, case_id)
            else:
                source = read_landmarks(rec.source_landmarks)
                if method == "initial":
                    xy = source
                elif method == "tiled":
                    t = find_transform(outdir, rec.patient_id, "tiled")
                    xy = warp_tiled(t, rec.moving_channels, source, stage=stage)
                else:
                    t = find_transform(outdir, rec.patient_id, "valis")
                    xy = warp_valis(t, rec.moving_stem, source)
                row["time_min"] = times.get(str(rec.patient_id), float("nan"))
            path = write_landmarks(out_dir / f"{case_id}.csv", xy)
            row["warped_path"] = str(path)
        except Exception as exc:  # one bad case must not sink the batch
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"case {case_id} [{method}]: {row['error']}", file=sys.stderr)
        rows.append(row)
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    index.to_csv(out_dir / "warp_index.csv", index=False)
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Warp ANHIR source landmarks through a method's transform."
    )
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--pairs", required=True, help="pairs_manifest.csv from prepare.py")
    ap.add_argument("--out", required=True, help="warped-landmark directory to write")
    ap.add_argument(
        "--outdir",
        help="the pipeline --outdir of the run for this method (tiled / valis)",
    )
    ap.add_argument("--trace", help="that run's -with-trace file, for execution time")
    ap.add_argument("--baseline-root", help="ANHIR's bUnwarpJ folder (method bunwarpj)")
    ap.add_argument(
        "--stage",
        default="refined",
        choices=("rigid", "refined"),
        help="STARE stage to warp through (tiled only)",
    )
    a = ap.parse_args(argv)
    pairs = load_pairs(a.pairs)
    index = warp_all(
        a.method,
        pairs,
        a.out,
        outdir=a.outdir,
        baseline_root=a.baseline_root,
        trace=a.trace,
        stage=a.stage,
    )
    n_bad = int((index["error"] != "").sum())
    print(f"{a.method}: {len(index) - n_bad}/{len(index)} cases warped -> {a.out}")
    return 1 if n_bad == len(index) and len(index) else 0


if __name__ == "__main__":
    sys.exit(main())
