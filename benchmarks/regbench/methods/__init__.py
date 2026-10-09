"""The methods, and the one driver that runs any of them on a case.

A method is a module with

``VARIANTS``                    the result names it produces, in the order it produces them
``register(case, work, opts)``  a generator yielding ``(variant, warp, info)``, where
                                ``warp(xy (N, 2)) -> (N, 2)`` maps MOVING-image pixels to
                                REFERENCE-image pixels (X, Y, full resolution)

It yields each variant as soon as that stage exists (rigid before non-rigid before micro), so
what is recorded for a variant is the cost of reaching it, and a later stage that crashes does
not lose the earlier ones. Methods import their software inside ``register``: each runs in its
own environment, and the scorer in none of them.

**What is billed.** Everything from the call to ``register`` until the variant is yielded:
reading the images, any conversion the method needs, the registration itself. Warping the
benchmark's points is the benchmark's work and is taken out (``warp_s`` records it), which is
also the challenge's rule ("time includes image loading, excludes landmark warping"). Peak
memory is the running maximum, so it cannot be un-counted; it is the peak up to that point.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

from .. import METHODS
from ..cases import load_points, result_dir, write_json, write_npz
from ..resources import Monitor, host_info


def load(method):
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; expected one of {METHODS}")
    return importlib.import_module(f"{__name__}.{method}")


def variant_names(method, label=None):
    """``--label`` renames a method's variants, so two configurations of it can sit side by side."""
    names = load(method).VARIANTS
    return [v.replace(method, label, 1) for v in names] if label else list(names)


def parse_opts(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--opt wants key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _warp_chunked(warp, xy, chunk=2_000_000):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return xy
    out = [np.asarray(warp(xy[i : i + chunk]), dtype=float).reshape(-1, 2)
           for i in range(0, len(xy), chunk)]
    out = np.concatenate(out)
    if out.shape != xy.shape:
        raise RuntimeError(f"warp returned {out.shape} for {xy.shape} points")
    return out


def jacobian_grid(hw, max_side=512):
    """``(xy (N, 2), (rows, cols), step)``: a regular lattice over the moving slide, at most
    ``max_side`` points a side. The scorer differentiates each method's map on it."""
    h, w = hw
    step = max(8, -(-max(h, w) // max_side))
    ys, xs = np.arange(step / 2.0, h, step), np.arange(step / 2.0, w, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1), (len(ys), len(xs)), step


def run_case(method, case, out, opts=None, label=None, workers=1, work=None, keep_work=False):
    """Register one case with one method; returns the list of variants that succeeded."""
    mod = load(method)
    opts = dict(opts or {})
    opts.setdefault("workers", workers)
    names = dict(zip(mod.VARIANTS, variant_names(method, label)))
    points = load_points(case)
    lm = points.get("landmarks_mov", np.empty((0, 2)))
    cx = points.get("cells_mov_xy", np.empty((0, 2)))
    grid, grid_shape, grid_step = (jacobian_grid(case.mov_hw) if case.mov_hw
                                   else (np.empty((0, 2)), (0, 0), 1))
    work = Path(work) if work else result_dir(out, names[mod.VARIANTS[-1]], case) / "work"
    work.mkdir(parents=True, exist_ok=True)

    base = {"method": method, "dataset": case.dataset, "case_id": case.case_id, "opts": opts,
            "workers": int(workers), "n_landmarks": len(lm), "n_cell_vertices": len(cx),
            **host_info()}
    # Written first: if the scheduler kills the task (out of memory, time limit) nothing below
    # runs, and this is what the scorer finds instead of silence.
    for variant in mod.VARIANTS:
        d = result_dir(out, names[variant], case)
        (d / "warped.npz").unlink(missing_ok=True)
        write_json(d / "run.json", {**base, "variant": names[variant], "ok": False,
                                    "error": "killed before finishing (out of memory or time "
                                             "limit? see the SLURM log and slurm_accounting.psv)"})
    done, error, tb = [], "", ""
    mon = Monitor().start()
    try:
        for variant, warp, info in mod.register(case, work, opts):
            cost = mon.snapshot()
            w0, c0 = mon.raw()
            arrays = {"landmarks": _warp_chunked(warp, lm),
                      "cells_xy": _warp_chunked(warp, cx).astype(np.float32),
                      "grid": _warp_chunked(warp, grid).astype(np.float32),
                      "grid_shape": np.array(grid_shape), "grid_step": np.array(grid_step)}
            w1, c1 = mon.raw()
            mon.exclude(w1 - w0, c1 - c0)
            d = result_dir(out, names[variant], case)
            write_npz(d / "warped.npz", **arrays)
            write_json(d / "run.json", {**base, "variant": names[variant], "ok": True, "error": "",
                                        "register_s": cost["wall_s"], "warp_s": w1 - w0,
                                        "resources": cost, "info": info})
            done.append(variant)
    except Exception as exc:  # a failed case is scored at the initial pose
        error, tb = f"{type(exc).__name__}: {exc}", traceback.format_exc()
        print(f"{case.dataset}/{case.case_id} [{method}]: {error}", file=sys.stderr)
    finally:
        cost = mon.snapshot()
        end = mon.stop()
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    for variant in mod.VARIANTS:
        d = result_dir(out, names[variant], case)
        if variant in done:
            # the scheduler's peak is only known once the task is over
            rec = json.loads((d / "run.json").read_text())
            rec["resources"].update(end)
            write_json(d / "run.json", rec)
        else:
            write_json(d / "run.json", {**base, "variant": names[variant], "ok": False,
                                        "error": error or "the method did not produce this variant",
                                        "traceback": tb, "register_s": cost["wall_s"],
                                        "resources": {**cost, **end}})
    return [names[v] for v in done]
