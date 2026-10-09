"""Known-truth error: how far a method's map leaves a point from where it truly belongs.

Synthetic and semi-synthetic cases are made by a known map ``g`` (moving -> reference,
``datasets/synthetic.forward_map``). Their ``points.npz`` holds paired probes::

    truth_mov (K, 2)    centroids of nuclei segmented on the native MOVING slide
    truth_ref (K, 2)    g(truth_mov), exactly
    truth_cell (K,)     which moving nucleus each probe is

**Direction.** A method is asked only for what it estimated. For a ``mov_to_ref`` map the
driver stores ``fn(truth_mov)`` and the error is its distance to ``truth_ref``; for a
``ref_to_mov`` map it stores ``fn(truth_ref)`` and the error is its distance to
``truth_mov``. The same pairs serve both directions. Neither the method's map nor ``g`` is
ever turned around, which also keeps the discontinuous ``seams`` family well defined.

**The frozen spec.** ``truth_spec.json`` fixes the deformation families, their amplitudes
and scales, the two poses, the window size, the probes and the thresholds. It carries the
SHA-256 of its own content; ``prepare`` refuses a file whose hash does not match and writes
the hash into every ``case.json`` it makes from it, and ``score`` refuses to mix cases of
two hashes. Families are reported one by one and never pooled.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SPEC_PATH = Path(__file__).with_name("truth_spec.json")
HASH_KEY = "sha256"
ERROR_KEYS = (
    "n_probes",
    "n_nonfinite",
    "tre_um_p50",
    "tre_um_p90",
    "tre_um_max",
    "frac_under_1um",
    "frac_under_2um",
    "initial_tre_um_p50",
    "frac_improved",
)
JACOBIAN_KEYS = ("fold_pct", "sd_log_jac")


# ── the frozen spec ───────────────────────────────────────────────────────────
def spec_hash(spec):
    """SHA-256 of the spec's content: canonical JSON of everything but the hash itself."""
    body = {k: v for k, v in spec.items() if k != HASH_KEY}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def load_spec(path=None):
    """The frozen spec, refused when its content does not match the hash it carries."""
    path = Path(path or SPEC_PATH)
    spec = json.loads(path.read_text())
    want, got = spec.get(HASH_KEY), spec_hash(spec)
    if want != got:
        raise SystemExit(
            f"{path}: the content hashes to {got} but the file says {want}. The "
            "known-truth spec is frozen; a change after any method has been scored is a "
            "post hoc change and needs a new hash, a new tag and a label in the results."
        )
    return spec


def family_names(spec):
    return [f["name"] for f in spec["families"]]


def pose_names(spec):
    return [p["name"] for p in spec["poses"]]


def family(spec, name):
    for f in spec["families"]:
        if f["name"] == name:
            return f
    raise KeyError(f"unknown family {name!r}; expected one of {family_names(spec)}")


def in_stare_model(spec, name):
    """False for the families a B-spline mesh on STARE's lattice cannot represent."""
    return bool(family(spec, name)["in_stare_model"])


def family_kwargs(spec, name, pixel_size_um):
    """The generator's keywords for one family. Lengths given in micrometres are turned
    into pixels by the case's pixel size, never by the window's side."""
    f = family(spec, name)
    kind = f["field"]
    if kind == "none":
        return dict(field="wave", amp=0.0)
    if kind == "wave":
        return dict(field="wave", amp=f["amp_px"], wavelength=f["wavelength_px"])
    if kind == "grid":
        return dict(field="grid", amp=0.0, grid_sd=f["sd_px"], grid_nodes=f["nodes"])
    if kind == "multiscale":
        px = float(pixel_size_um)
        scales = tuple(s / px for s in f["scales_um"])
        # equal energy at every scale and in both components: the RMS of |w| is rms_um
        rms = f["rms_um"] / px / math.sqrt(2.0 * len(scales))
        return dict(
            field="multiscale", amp=0.0, scales=scales, rms=(rms,) * len(scales)
        )
    if kind == "bumps":
        return dict(
            field="bumps", amp=0.0, bump_sigma=f["sigma_px"], bump_count=f["count"]
        )
    if kind == "seams":
        return dict(field="seams", amp=0.0, seam_tile=f["block_px"], seam_sd=f["sd_px"])
    raise ValueError(f"family {name!r}: unknown field {kind!r}")


def pose_kwargs(spec, name):
    for p in spec["poses"]:
        if p["name"] == name:
            return dict(theta=p["theta_deg"], shift=tuple(p["shift_px"]))
    raise KeyError(f"unknown pose {name!r}; expected one of {pose_names(spec)}")


def case_extra(spec, family_name, frozen):
    """What a known-truth case records about the spec. The hash is written only for a case
    made exactly as the spec says (``frozen``); a reduced copy for a test carries none."""
    return {
        "in_stare_model": in_stare_model(spec, family_name),
        "truth_spec_sha256": spec[HASH_KEY] if frozen else "",
    }


# ── probes ────────────────────────────────────────────────────────────────────
def probe_arrays(centroids, g, ref_hw, seed, spec=None):
    """``truth_mov``, ``truth_ref``, ``truth_cell`` from the moving nuclei's centroids.

    A probe is kept when its true position lies at least ``margin_px`` inside the
    reference; at most ``max`` are kept, by a seeded draw, in their original order.
    """
    spec = spec or load_spec()
    cap, margin = int(spec["probes"]["max"]), float(spec["probes"]["margin_px"])
    cent = np.asarray(centroids, dtype=float).reshape(-1, 2)
    idx = np.flatnonzero(np.isfinite(cent).all(axis=1))
    target = np.asarray(g(cent[idx]), dtype=float).reshape(-1, 2)
    h, w = ref_hw
    ok = (
        np.isfinite(target).all(axis=1)
        & (target >= margin).all(axis=1)
        & (target[:, 0] <= w - 1 - margin)
        & (target[:, 1] <= h - 1 - margin)
    )
    idx, target = idx[ok], target[ok]
    if len(idx) > cap:
        rng = np.random.default_rng(int(seed) + int(spec["probes"]["seed_offset"]))
        keep = np.sort(rng.choice(len(idx), cap, replace=False))
        idx, target = idx[keep], target[keep]
    return {
        "truth_mov": cent[idx],
        "truth_ref": target,
        "truth_cell": idx.astype(np.int64),
    }


# ── the error ─────────────────────────────────────────────────────────────────
def _percentile(sorted_values, q):
    """Linear percentile of an ascending array that may end in ``inf``."""
    n = len(sorted_values)
    pos = q / 100.0 * (n - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    a, b = float(sorted_values[lo]), float(sorted_values[hi])
    if lo == hi or a == b:
        return a
    if math.isinf(b):
        return math.inf
    return a + (b - a) * (pos - lo)


def error_stats(mapped, target, source, pixel_size_um):
    """Error of ``mapped = fn(source)`` against ``target``, in micrometres.

    A point the method returned as NaN or infinite counts AGAINST it: its error is
    infinite, so it is under no threshold, it is not improved, and it pulls the percentiles
    up. Without a pixel size the micrometre numbers are NaN, never zero.
    """
    out = dict.fromkeys(ERROR_KEYS, math.nan)
    mapped = np.asarray(mapped, dtype=float).reshape(-1, 2)
    target = np.asarray(target, dtype=float).reshape(-1, 2)
    source = np.asarray(source, dtype=float).reshape(-1, 2)
    out["n_probes"], out["n_nonfinite"] = len(target), 0
    if not len(target) or len(mapped) != len(target):
        return out
    bad = ~np.isfinite(mapped).all(axis=1)
    err = np.full(len(target), math.inf)
    err[~bad] = np.hypot(*(mapped[~bad] - target[~bad]).T)
    init = np.hypot(*(source - target).T)
    out["n_nonfinite"] = int(bad.sum())
    out["frac_improved"] = float(np.mean(err < init))
    if not pixel_size_um:
        return out
    px = float(pixel_size_um)
    err_um, init_um = np.sort(err * px), np.sort(init * px)
    out["tre_um_p50"] = _percentile(err_um, 50)
    out["tre_um_p90"] = _percentile(err_um, 90)
    out["tre_um_max"] = float(err_um[-1])
    out["frac_under_1um"] = float(np.mean(err_um < 1.0))
    out["frac_under_2um"] = float(np.mean(err_um < 2.0))
    out["initial_tre_um_p50"] = _percentile(init_um, 50)
    return out


def jacobian_stats(warped):
    """``fold_pct`` and ``sd_log_jac`` of a method's map, from the lattice the driver pushed
    through it (``warped.npz``: ``grid``, ``grid_shape``, ``grid_step``). Forward evaluation
    only: the map is differentiated where the method defined it."""
    out = dict.fromkeys(JACOBIAN_KEYS, math.nan)
    if warped is None or "grid" not in warped or min(warped["grid_shape"]) < 3:
        return out
    rows, cols = (int(v) for v in warped["grid_shape"])
    step = float(warped["grid_step"])
    g = np.asarray(warped["grid"], dtype=float).reshape(rows, cols, 2)
    jac = np.gradient(g[..., 0], step, axis=1) * np.gradient(
        g[..., 1], step, axis=0
    ) - np.gradient(g[..., 0], step, axis=0) * np.gradient(g[..., 1], step, axis=1)
    jac = jac[np.isfinite(jac)]
    if len(jac):
        out["fold_pct"] = float(100.0 * np.mean(jac <= 0))
        if np.any(jac > 0):
            out["sd_log_jac"] = float(np.std(np.log(jac[jac > 0])))
    return out


# ── per family ────────────────────────────────────────────────────────────────
def _agg(rows, key, fn):
    """``fn`` over a column, NaN left out and ``inf`` kept (a lost point is not hidden)."""
    v = np.array([float(r[key]) for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    return float(fn(v)) if v.size else math.nan


def by_family(rows):
    """One row per (dataset, family, pose, method). There is NO row over families: a
    method that cannot represent a family must show it in that family's own row."""
    out = []
    keys = dict.fromkeys(
        (r["dataset"], r["family"], r["pose"], r["method"]) for r in rows
    )
    for ds, fam, pose, method in keys:
        mine = [
            r
            for r in rows
            if (r["dataset"], r["family"], r["pose"], r["method"])
            == (ds, fam, pose, method)
        ]
        out.append(
            {
                "dataset": ds,
                "family": fam,
                "pose": pose,
                "in_stare_model": mine[0]["in_stare_model"],
                "method": method,
                "n_cases": len(mine),
                "n_ok": sum(not r["imputed_initial"] for r in mine),
                "tre_um_p50_med": _agg(mine, "tre_um_p50", np.median),
                "tre_um_p50_max": _agg(mine, "tre_um_p50", np.max),
                "tre_um_p90_med": _agg(mine, "tre_um_p90", np.median),
                "tre_um_max_max": _agg(mine, "tre_um_max", np.max),
                "frac_under_1um_med": _agg(mine, "frac_under_1um", np.median),
                "frac_under_2um_med": _agg(mine, "frac_under_2um", np.median),
                "initial_tre_um_p50_med": _agg(mine, "initial_tre_um_p50", np.median),
                "fold_pct_med": _agg(mine, "fold_pct", np.median),
            }
        )
    return out
