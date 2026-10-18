"""The paper's cost figures (Task 31): fig. 3e, 3f, 2c and the supplementary CPU cross
and end-to-end panels, every one by the rules of ``paper.costs``.

* **One unit per axis.** Fig. 3e and 3f: one point per PATIENT for every method. Fig. 2c:
  a per-round facet (every method that has a per-round figure) and a per-patient facet
  (every method, with the fits). The source data names the unit of every value.
* **The common patients** (``cost_points.in_common``) are what is drawn; every tick
  reads ``n=<common>/<all> (capped c, failed f)``, and ``+k not drawn`` when values of
  common patients could not be (no figure, or <= 0 on a log axis).
* **Rows.** Each family's PRIMARY row filled; its best-Dice row hollow, labelled "best
  (ceiling)". A GPU row is never on a CPU-hours or a 4-CPU wall axis: there it is its
  CPU twin (``variants.cpu_twin``), and its GPU-hours have a panel of their own,
  labelled with the GPU model the points carry (``cost_points.device`` /
  ``gpu_model``, so a draft whose device reads ``cpu|gpu`` is read the same way).
* **Clocks.** The wall panel of fig. 3f is every method at 4 CPUs (``whole_process``),
  STARE as ``stare_local4`` (``local_4cpu``); STARE's staged critical path and its time
  with the queue are two extra markers, each labelled.
* **Intervals.** A median's bootstrap 95% interval over the drawn patients
  (``costs.interval``: ``score.bootstrap_interval``) is drawn as a bar beside it.
* **Nothing silent.** Every point not drawn, every lower-bound point and every
  placeholder of these figures is a row of ``<figures>/losses.csv`` (stage ``paper``,
  ``write_losses``).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.lines
import matplotlib.ticker
import numpy as np

from .. import losses as losses_mod
from .. import variants
from . import costs
from . import plots as P
from . import plotstyle as ps

LOG_FMT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
CEILING = "best (ceiling)"
GPU, CPU = "gpu", "cpu"
SOURCE = "paper/cost_points.csv"


# ── reading the points ───────────────────────────────────────────────────────
def _b(v):
    """A boolean cell (None when the table has no value: a draft)."""
    if v is None or v == "" or v == "NA":
        return None
    return P._flag(v)


def _rows_to_draw(cost, best):
    """``[(family, row, kind)]``: per family its primary row then its best row (when
    another), from the points' own flags; for a table without them (a draft) the best
    rows of ``method_best``."""
    out, seen = [], set()
    flagged = any(_b(p.get("is_primary")) is not None for p in cost)
    fams = []
    for p in cost:
        if p.get("family") and p["family"] not in fams:
            fams.append(p["family"])
    for b in best or []:
        if b.get("family") and b["family"] not in fams:
            fams.append(b["family"])
    for fam in fams:
        mine = [p for p in cost if p.get("family") == fam]
        prim = [p["row"] for p in mine if _b(p.get("is_primary"))]
        bests = [p["row"] for p in mine if _b(p.get("is_best"))]
        if not flagged:
            bests = [b["best_row"] for b in best or [] if b.get("family") == fam and b.get("best_row")]
        for row in dict.fromkeys(prim):
            if (fam, row) not in seen:
                out.append((fam, row, costs.PRIMARY))
                seen.add((fam, row))
        for row in dict.fromkeys(bests):
            if (fam, row) not in seen:
                out.append((fam, row, costs.BEST))
                seen.add((fam, row))
    return out


def _patients(cost, row):
    return [p for p in cost if p.get("row") == row and (p.get("unit_kind") or "patient") == "patient"]


def _device(pts):
    devs = {str(p.get("device") or "") for p in pts} - {""}
    return GPU if any(GPU in d for d in devs) else CPU


def _gpu_model(pts):
    return "|".join(sorted({str(p.get("gpu_model")) for p in pts if p.get("gpu_model")})) or "GPU"


def _twin(cost, row):
    """A GPU row's CPU twin (``variants.cpu_twin``) when the points have it, or None."""
    twin = variants.cpu_twin(row)
    return twin if twin and _patients(cost, twin) else None


def _palette(fig_dir, cost):
    """The locked palette, with any family the points name that it does not know."""
    return P._colors(fig_dir, extra=sorted({p.get("family") for p in cost if p.get("family")}))


def _status(p):
    return p.get("status") or costs.COMPLETE


def _common(p):
    c = _b(p.get("in_common"))
    return True if c is None else c


def _comparable(p):
    return _b(p.get("peak_rss_comparable")) is not False


def _group(cost, row, key, log=True, peaks=False):
    """``(values drawn, tick counts, not drawn, lower-bound points)`` of one row on one
    quantity: the common patients that completed (rule 2)."""
    pts = _patients(cost, row)
    shared = [p for p in pts if _status(p) == costs.COMPLETE and _common(p)]
    drawn, gone, low = [], [], []
    for p in shared:
        v = P._num(p.get(key))
        if peaks and not _comparable(p):
            gone.append((p, "a rusage peak (a lower bound) is not compared"))
            continue
        if not math.isfinite(v) or (log and v <= 0):
            gone.append((p, "no figure" if not math.isfinite(v) else "<= 0 on a log axis"))
            continue
        drawn.append((p, v))
        if p.get("lower_bound"):
            low.append(p)
    counts = {
        "common": len(shared),
        "all": len(pts),
        "capped": sum(_status(p) == costs.CAPPED for p in pts),
        "failed": sum(_status(p) == costs.FAILED for p in pts),
        "not_drawn": len(gone),
    }
    return drawn, counts, gone, low


def tick(label, counts):
    """The tick of one group (rule 2)."""
    t = f"{label}\nn={counts['common']}/{counts['all']} (capped {counts['capped']}, failed {counts['failed']})"
    if counts["not_drawn"]:
        t += f"\n+{counts['not_drawn']} not drawn"
    return t


def _ci_bar(ax, pos, values, color):
    low, high = costs.interval(values)
    if math.isfinite(low) and math.isfinite(high):
        ax.plot([pos + 0.34, pos + 0.34], [low, high], color=color, lw=1.0, zorder=5)
    return low, high


def _loss(fig, panel, p, cause, effect, why):
    return losses_mod.row(
        stage="paper",
        dataset=p.get("dataset") or "",
        patient_id=p.get("unit_id") if (p.get("unit_kind") or "patient") == "patient" else "",
        case_id=p.get("unit_id") if p.get("unit_kind") == "round" else "",
        run=p.get("row") or "",
        cause=cause,
        effect=effect,
        detail=f"{fig} ({panel}): {why}",
        source=SOURCE,
    )


def _losses(fig, gone, low, panel):
    out = [_loss(fig, panel, p, "plot_points_dropped", "dropped", why) for p, why in gone]
    out += [
        _loss(fig, panel, p, "plot_lower_bound", "flagged", f"drawn, a lower bound ({p.get('lower_bound')})")
        for p in low
    ]
    return out


def _log_axis(ax, ylabel):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LOG_FMT)
    ax.yaxis.set_minor_formatter(LOG_FMT)
    ax.set_ylabel(ylabel)


def _label(fam, kind, extra=""):
    lab = P._label(fam) + extra
    return f"{lab}\n{CEILING}" if kind == costs.BEST else lab


COLS = ["panel", "quantity", "unit", "clock", "family", "row", "row_kind", "unit_id",
        "value", "ci_low", "ci_high", "lower_bound"]


def _box_panel(ax, cost, fig_name, panel, clock, entries, key, colors, ylabel, peaks=False):
    """One panel of boxes: ``entries`` = ``[(family, row, kind, label extra)]``."""
    labels, data, losses, n = [], [], [], {}
    for i, (fam, row, kind, extra) in enumerate(entries):
        drawn, counts, gone, low = _group(cost, row, key, peaks=peaks)
        vals = [v for _, v in drawn]
        P._draw_group(ax, i, vals, colors[fam], seed=i, hollow=kind == costs.BEST)
        low_ci, high_ci = _ci_bar(ax, i, vals, colors[fam]) if vals else (math.nan, math.nan)
        labels.append(tick(_label(fam, kind, extra), counts))
        n[f"{fam}:{kind}"] = counts["common"]
        losses += _losses(fig_name, gone, low, panel)
        for p, v in drawn:
            data.append({"panel": panel, "quantity": key, "unit": "patient",
                         "clock": p.get("clock") or clock, "family": fam, "row": row,
                         "row_kind": kind, "unit_id": p.get("unit_id"), "value": v,
                         "ci_low": low_ci, "ci_high": high_ci,
                         "lower_bound": p.get("lower_bound") or ""})
    P._group_axes(ax, labels)
    _log_axis(ax, ylabel)
    return data, losses, n


def _setup(paper_dir, fig_dir, name):
    got, missing = P._need(paper_dir, best="method_best.csv", cost="cost_points.csv")
    if missing:
        return None, P._placeholder(fig_dir, name, missing)
    return got, None


def _done(fig, fig_dir, name, data, n, losses, cols=COLS):
    files = P._save(fig, fig_dir, name, P._Table(cols, data))
    return P._result(files, n, losses=losses)


def _rotate(axes):
    for ax in axes:
        for t in ax.get_xticklabels():
            t.set_rotation(90)
            t.set_ha("center")


# ── Fig. 3e: the memory one job needs ────────────────────────────────────────
def fig3e_peak_rss_best(paper_dir, fig_dir):
    name = "fig3e_peak_rss_best"
    got, ph = _setup(paper_dir, fig_dir, name)
    if ph:
        return ph
    cost = got["cost"]
    entries = []
    for fam, row, kind in _rows_to_draw(cost, got["best"]):
        staged = any(p.get("clock") == costs.CRITICAL for p in _patients(cost, row))
        entries.append((fam, row, kind, " *" if staged else ""))
    P._style()
    colors = _palette(fig_dir, cost)
    fig, (ax,) = ps.figure(width="double", aspect=0.6)
    ylab = "Memory one job needs (GB, log10), per patient"
    if any(e[3] for e in entries):
        ylab += "\n* STARE: its largest stage job"
    data, losses, n = _box_panel(ax, cost, name, "peak", "", entries, "peak_rss_gb",
                                 colors, ylab, peaks=True)
    # STARE's concurrent memory: a second, labelled marker (R4, P5)
    marked = False
    for i, (fam, row, kind, star) in enumerate(entries):
        if not star:
            continue
        drawn, _, gone, _ = _group(cost, row, "peak_concurrent_gb", peaks=True)
        losses += _losses(name, gone, [], "peak_concurrent")
        vals = [v for _, v in drawn]
        if not vals:
            continue
        ax.scatter([i + 0.22], [float(np.median(vals))], marker="D", s=14, facecolor="none",
                   edgecolor=colors[fam], zorder=6,
                   label=None if marked else "STARE: memory its jobs of one round held at once (median)")
        marked = True
        for p, v in drawn:
            data.append({"panel": "peak", "quantity": "peak_concurrent_gb", "unit": "patient",
                         "clock": "", "family": fam, "row": row, "row_kind": kind,
                         "unit_id": p.get("unit_id"), "value": v, "ci_low": None,
                         "ci_high": None, "lower_bound": ""})
    if marked:
        ax.legend(loc="upper left", fontsize=ps.TICK_PT)
    return _done(fig, fig_dir, name, data, n, losses)


# ── Fig. 3f: what the cluster paid ───────────────────────────────────────────
def _cpu_entries(cost, best, local=False):
    """The CPU rows of the families: a GPU row is replaced by its CPU twin (or left
    out, counted, when it has none); with ``local``, STARE's staged row by
    ``stare_local4``."""
    out, left_out = [], []
    for fam, row, kind in _rows_to_draw(cost, best):
        pts = _patients(cost, row)
        if _device(pts) == GPU:
            twin = _twin(cost, row)
            if twin is None:
                left_out.append((fam, row, kind, "a GPU row with no CPU twin: not on a CPU axis"))
                continue
            out.append((fam, twin, kind, " (CPU twin)"))
            continue
        if local and any(p.get("clock") == costs.CRITICAL for p in pts):
            loc = [r for r in variants.LOCAL_4CPU if _patients(cost, r)]
            if loc:
                out.append((fam, loc[0], kind, " (4 CPUs, one job)"))
                continue
            left_out.append((fam, row, kind, "STARE's stage jobs with no stare_local4: not on the 4-CPU axis"))
            continue
        out.append((fam, row, kind, ""))
    return out, left_out


def _bars(ax, i, used, reserved, total, color, hollow):
    face = "none" if hollow else color
    if math.isfinite(reserved):
        ax.bar(i, reserved, width=0.6, color=color, alpha=0.25, edgecolor=color, lw=0.6, zorder=1)
    if math.isfinite(used):
        ax.bar(i, used, width=0.3, color=face, edgecolor=color, lw=0.8, zorder=2)
    if math.isfinite(total):
        ax.scatter([i], [total], marker="^", s=14, color=color, zorder=4)


def _patient_rows(drawn, key_values):
    """The source-data rows of the drawn patients, one per quantity of ``key_values``."""
    out = []
    for p, v in drawn:
        for q in key_values:
            out.append((p, q, v if q == key_values[0] else P._num(p.get(q))))
    return out


def fig3f_cpu_wall_best(paper_dir, fig_dir):
    name = "fig3f_cpu_wall_best"
    got, ph = _setup(paper_dir, fig_dir, name)
    if ph:
        return ph
    cost, best = got["cost"], got["best"]
    P._style()
    colors = _palette(fig_dir, cost)
    fig, axes = ps.figure(width="double", nrows=2, ncols=2, aspect=1.1)
    axes = list(np.ravel(axes))
    data, losses, n = [], [], {}
    cpu_rows, gpu_left = _cpu_entries(cost, best)

    def put(panel, fam, row, kind, rows):
        for p, q, val in rows:
            data.append({"panel": panel, "quantity": q, "unit": "patient",
                         "clock": p.get("clock"), "family": fam, "row": row,
                         "row_kind": kind, "unit_id": p.get("unit_id"), "value": val,
                         "ci_low": None, "ci_high": None,
                         "lower_bound": p.get("lower_bound") or ""})

    # (a) CPU-hours: reserved bar, used inside it, total (good + wasted) marker (R8, R10)
    labels = []
    for i, (fam, row, kind, extra) in enumerate(cpu_rows):
        drawn, counts, gone, low = _group(cost, row, "cpu_h_used")
        losses += _losses(name, gone, low, "cpu_h")
        pts = [p for p, _ in drawn]
        used = [v for _, v in drawn]
        _bars(axes[0], i, float(np.median(used)) if used else math.nan,
              costs.median(p.get("cpu_h_reserved") for p in pts),
              costs.median(p.get("total_cpu_h_reserved") for p in pts), colors[fam],
              kind == costs.BEST)
        _ci_bar(axes[0], i, used, colors[fam])
        labels.append(tick(_label(fam, kind, extra), counts))
        n[f"cpu_h:{fam}:{kind}"] = counts["common"]
        put("cpu_h", fam, row, kind,
            _patient_rows(drawn, ("cpu_h_used", "cpu_h_reserved", "total_cpu_h_reserved")))
    P._group_axes(axes[0], labels)
    _log_axis(axes[0], "CPU-hours per patient (log10)\nbar: reserved; inside: used; ^: total "
                       "with wasted attempts")
    # (b) GB-hours: used, the wasted request stacked on it; the request as a line
    labels = []
    for i, (fam, row, kind, extra) in enumerate(cpu_rows):
        drawn, counts, gone, low = _group(cost, row, "gb_h_used")
        losses += _losses(name, gone, low, "gb_h")
        pts = [p for p, _ in drawn]
        used = float(np.median([v for _, v in drawn])) if drawn else math.nan
        wasted = costs.median(p.get("wasted_gb_h_reserved") for p in pts)
        asked = costs.median(p.get("gb_h_reserved") for p in pts)
        face = "none" if kind == costs.BEST else colors[fam]
        if math.isfinite(used):
            axes[1].bar(i, used, width=0.5, color=face, edgecolor=colors[fam], lw=0.8)
            if math.isfinite(wasted) and wasted > 0:
                axes[1].bar(i, wasted, bottom=used, width=0.5, color=colors[fam], alpha=0.3,
                            hatch="//", edgecolor=colors[fam], lw=0.5)
        if math.isfinite(asked):
            axes[1].plot([i - 0.3, i + 0.3], [asked, asked], color=ps.GREY, lw=1.0)
        labels.append(tick(_label(fam, kind, extra), counts))
        n[f"gb_h:{fam}:{kind}"] = counts["common"]
        put("gb_h", fam, row, kind,
            _patient_rows(drawn, ("gb_h_used", "wasted_gb_h_reserved", "gb_h_reserved")))
    P._group_axes(axes[1], labels)
    _log_axis(axes[1], "GB-hours per patient (log10)\nbar: used (peak x time); hatched: "
                       "wasted request;\nline: the request (STARE: the pipeline's, others: "
                       "the 16 GB-floor ladder)")
    # (c) GPU-hours, GPU rows only (R11, P7)
    gpu_rows = [(f, r, k) for f, r, k in _rows_to_draw(cost, best) if _device(_patients(cost, r)) == GPU]
    if gpu_rows:
        entries = [(f, r, k, f" ({_gpu_model(_patients(cost, r))})") for f, r, k in gpu_rows]
        d, lo, nn = _box_panel(axes[2], cost, name, "gpu_h", costs.WHOLE, entries,
                               "gpu_h", colors, "GPU-hours per patient (log10)")
        data += d
        losses += lo
        n.update({f"gpu_h:{k}": v for k, v in nn.items()})
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "no GPU row", ha="center", va="center", transform=axes[2].transAxes,
                     fontsize=ps.TICK_PT)
    # (d) wall time, every method at 4 CPUs; STARE as stare_local4, plus its two markers
    wall_rows, wall_left = _cpu_entries(cost, best, local=True)
    d, lo, nn = _box_panel(axes[3], cost, name, "wall", costs.WHOLE, wall_rows, "wall_s",
                           colors, "Wall time at 4 CPUs (s, log10), per patient\nshared nodes")
    data += d
    losses += lo
    n.update({f"wall:{k}": v for k, v in nn.items()})
    for i, (fam, row, kind, extra) in enumerate(wall_rows):
        if row not in variants.LOCAL_4CPU:
            continue
        staged = [r for f, r, k in _rows_to_draw(cost, best)
                  if f == fam and k == kind and any(p.get("clock") == costs.CRITICAL for p in _patients(cost, r))]
        for r in staged:
            for key, mark, text in (("wall_s", "s", "STARE stage jobs: critical path, every tile at once"),
                                    ("elapsed_with_queue_s", "v", "STARE stage jobs: with the queue")):
                drawn, _, _, _ = _group(cost, r, key)
                vals = [v for _, v in drawn]
                if not vals:
                    continue
                axes[3].scatter([i + 0.25], [float(np.median(vals))], marker=mark, s=16,
                                facecolor="none", edgecolor=colors[fam], zorder=6, label=text)
                for p, v in drawn:
                    data.append({"panel": "wall", "quantity": key, "unit": "patient",
                                 "clock": costs.QUEUE if key == "elapsed_with_queue_s" else costs.CRITICAL,
                                 "family": fam, "row": r, "row_kind": kind, "unit_id": p.get("unit_id"),
                                 "value": v, "ci_low": None, "ci_high": None, "lower_bound": ""})
    handles, texts = axes[3].get_legend_handles_labels()
    if handles:
        seen = dict(zip(texts, handles))
        axes[3].legend(seen.values(), seen.keys(), loc="upper left", fontsize=ps.TICK_PT)
    _rotate(axes)
    for fam, row, kind, why in gpu_left + wall_left:
        for p in _patients(cost, row):
            if _status(p) == costs.COMPLETE and _common(p):
                losses.append(_loss(name, "cpu_h|gb_h|wall", p, "plot_points_dropped", "dropped", why))
    ps.panel_labels(axes)
    return _done(fig, fig_dir, name, data, n, losses)


# ── Fig. 2c: scaling with the input ──────────────────────────────────────────
def fig2c_cost_vs_input(paper_dir, fig_dir):
    name = "fig2c_cost_vs_input"
    got, missing = P._need(paper_dir, best="method_best.csv", cost="cost_points.csv",
                           fits="cost_fits.csv")
    if missing:
        return P._placeholder(fig_dir, name, missing)
    cost = got["cost"]
    P._style()
    colors = _palette(fig_dir, cost)
    fig, axes = ps.figure(width="double", nrows=2, ncols=2, aspect=0.95)
    axes = np.asarray(axes).reshape(2, 2)
    rows = _rows_to_draw(cost, got["best"])
    data, losses, n = [], [], {}
    for r_i, unit in enumerate(("round", "patient")):
        for c_i, (key, ylab) in enumerate((("peak_rss_gb", "Peak memory (GB)"),
                                            ("cpu_h_used", "CPU-hours"))):
            ax = axes[r_i, c_i]
            for fam, row, kind in rows:
                pts = [p for p in cost if p.get("row") == row
                       and (p.get("unit_kind") or "patient") == unit]
                if unit == "patient":
                    pts = [p for p in pts if _status(p) == costs.COMPLETE]
                good, gone = [], []
                for p in pts:
                    x, y = P._num(p.get("input_gpx")), P._num(p.get(key))
                    if key == "peak_rss_gb" and not _comparable(p):
                        gone.append((p, "a rusage peak (a lower bound) is not compared"))
                    elif math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0:
                        good.append((p, x, y))
                    else:
                        gone.append((p, "no size or value, or <= 0 on a log axis"))
                losses += _losses(name, gone, [], f"{unit}:{key}")
                if not good:
                    continue
                xs = np.array([g[1] for g in good])
                ys = np.array([g[2] for g in good])
                hollow = kind == costs.BEST
                ax.scatter(xs, ys, s=7, facecolor="none" if hollow else colors[fam],
                           edgecolor=colors[fam], linewidths=0.6,
                           label=P._label(fam) + (f" {CEILING}" if hollow else ""))
                n[f"{unit}:{key}:{fam}:{kind}"] = len(good)
                for p, x, y in good:
                    data.append({"panel": f"{unit}:{key}", "unit": unit, "family": fam, "row": row,
                                 "row_kind": kind, "unit_id": p.get("unit_id"), "input_gpx": x,
                                 "value": y, "fit": False})
                if unit != "patient":
                    continue
                for f in got["fits"]:
                    if (f.get("row") == row and f.get("target") == key
                            and f.get("predictor") == "input_gpx" and P._flag(f.get("fit_ok"))
                            and P._fin(f.get("slope"))):
                        lx = np.array([xs.min(), xs.max()])
                        ly = 10 ** (P._num(f["intercept"]) + P._num(f["slope"]) * np.log10(lx))
                        ax.plot(lx, ly, color=colors[fam], lw=0.8, ls="--" if hollow else "-")
                        for a, b in zip(lx, ly):
                            data.append({"panel": f"{unit}:{key}", "unit": unit, "family": fam,
                                         "row": row, "row_kind": kind, "unit_id": None,
                                         "input_gpx": a, "value": b, "fit": True})
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(LOG_FMT)
            ax.yaxis.set_major_formatter(LOG_FMT)
            ax.set_xlabel("Moving image (gigapixels, log10)")
            ax.set_ylabel(f"{ylab}, per {unit} (log10)")
            ax.set_title("per round (pair methods; a per-patient method only where it timed "
                         "each round)" if unit == "round" else "per patient (every method; "
                         "line: log-log fit, >= 5 patients)", fontsize=ps.TICK_PT)
    ps.panel_labels(list(axes.ravel()))
    handles, labels = axes[1, 0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fits_ok = [f for f in got["fits"] if P._flag(f.get("fit_ok"))]
    short = [f for f in got["fits"] if not P._flag(f.get("fit_ok")) and f.get("predictor") == "input_gpx"]
    if seen:
        fig.legend(seen.values(), seen.keys(), loc="outside lower center", ncol=min(4, len(seen)),
                   title=f"Method (filled: primary; hollow: {CEILING}); {len(fits_ok)} fit(s) "
                         f"drawn, {len(short)} not (fewer than 5 patients or 2 sizes)")
    return P._result(
        P._save(fig, fig_dir, name, P._Table(["panel", "unit", "family", "row", "row_kind",
                                               "unit_id", "input_gpx", "value", "fit"], data)),
        n, losses=losses)


# ── supplementary: the CPU cross ─────────────────────────────────────────────
def figS_cpu_scaling(paper_dir, fig_dir):
    name = "figS_cpu_scaling"
    got, missing = P._need(paper_dir, cs="cpu_scaling.csv")
    if missing:
        return P._placeholder(fig_dir, name, missing)
    P._style()
    fams = sorted({r.get("family") for r in got["cs"] if r.get("family")})
    colors = P._colors(fig_dir, extra=fams)
    fig, axes = ps.figure(width="double", nrows=1, ncols=3, aspect=0.45)
    data, n, losses = [], {}, []
    for ax, key, ylab in ((axes[0], "speedup_vs_4", "Speed-up against 4 CPUs"),
                          (axes[1], "cpu_h_used", "CPU-hours per patient"),
                          (axes[2], "peak_rss_gb", "Peak memory (GB) per patient")):
        lines = {}
        for r in got["cs"]:
            v, c = P._num(r.get(key)), P._num(r.get("cpus"))
            if not (math.isfinite(v) and math.isfinite(c)):
                losses.append(losses_mod.row(
                    stage="paper", dataset=r.get("dataset") or "", patient_id=r.get("patient_id") or "",
                    run=r.get("row") or "", cause="plot_points_dropped", effect="dropped",
                    detail=f"{name} ({key}): no figure", source="paper/cpu_scaling.csv"))
                continue
            fam = r.get("family") or "stare"
            lines.setdefault((fam, r["row"], r.get("patient_id")), {}).setdefault(c, []).append(v)
        for (fam, row, pid), by in lines.items():
            cs = sorted(by)
            med = [float(np.median(by[c])) for c in cs]
            lo = [float(np.min(by[c])) for c in cs]
            hi = [float(np.max(by[c])) for c in cs]
            ax.plot(cs, med, color=colors.get(fam, ps.GREY), lw=0.6, marker="o", ms=2)
            ax.vlines(cs, lo, hi, color=colors.get(fam, ps.GREY), lw=0.5)
            n[f"{key}:{row}"] = n.get(f"{key}:{row}", 0) + 1
            for c, m, a, b in zip(cs, med, lo, hi):
                data.append({"panel": key, "family": fam, "row": row, "patient_id": pid,
                             "cpus": c, "median_over_repeats": m, "min": a, "max": b,
                             "n_repeats": len(by[c])})
        if key == "speedup_vs_4" and data:
            cs = sorted({d["cpus"] for d in data})
            ax.plot(cs, [c / 4.0 for c in cs], color=ps.LIGHT_GREY, lw=0.8, ls=":", label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({d["cpus"] for d in data}) or [4, 8])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("CPUs (one job, log2)")
        ax.set_ylabel(ylab)
    fig.legend(handles=[matplotlib.lines.Line2D([], [], color=colors.get(f, ps.GREY), label=P._label(f))
                        for f in fams],
               title="Best-Dice row of each family (CPU rows only);\none line per patient, "
                     "bar: the spread over repeats", loc="outside right center")
    ps.panel_labels(list(axes))
    files = P._save(fig, fig_dir, name, P._Table(
        ["panel", "family", "row", "patient_id", "cpus", "median_over_repeats", "min", "max",
         "n_repeats"], data))
    return P._result(files, n, losses=losses)


# ── supplementary: end to end ────────────────────────────────────────────────
PARTS = (("convert_s_med", "the method's conversion", "//"),
         ("estimate_s_med", "registration", ""),
         ("write_s_med", "the method's writer", ".."),
         ("prior_s_med", "the anchor it reads (affine ASHLAR)", "xx"))


def figS_end_to_end(paper_dir, fig_dir):
    name = "figS_end_to_end"
    got, missing = P._need(paper_dir, e2e="end_to_end.csv")
    if missing:
        return P._placeholder(fig_dir, name, missing)
    rows = [r for r in got["e2e"] if any(P._fin(r.get(k)) for k, _, _ in PARTS)]
    if not rows:
        return P._placeholder(fig_dir, name, [("end_to_end (no row with a figure)",
                                               Path(paper_dir) / "end_to_end.csv")])
    P._style()
    colors = _palette(fig_dir, rows)
    fig, (ax,) = ps.figure(width="double", aspect=0.5)
    labels, data = [], []
    for i, r in enumerate(rows):
        left = 0.0
        col = colors.get(r.get("family"), ps.GREY)
        for key, text, hatch in PARTS:
            v = P._num(r.get(key))
            if not math.isfinite(v) or v <= 0:
                continue
            ax.barh(i, v / 60.0, left=left, color=col, alpha=0.35 if hatch else 0.8,
                    hatch=hatch or None, edgecolor=col, lw=0.5, label=text if i == 0 else None)
            data.append({"row": r.get("row"), "family": r.get("family"), "part": key,
                         "minutes": v / 60.0, "unit": "patient",
                         "n_patients": r.get("n_patients")})
            left += v / 60.0
        labels.append(f"{P._label(r.get('family'))}: {r.get('row')}\n"
                      f"n={r.get('n_patients')} common patients")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel("Minutes per patient, median over the common patients (each part's own "
                  "median, stacked)")
    handles, texts = ax.get_legend_handles_labels()
    if handles:
        seen = dict(zip(texts, handles))
        ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=ps.TICK_PT)
    files = P._save(fig, fig_dir, name, P._Table(
        ["row", "family", "part", "minutes", "unit", "n_patients"], data))
    return P._result(files, {r.get("row"): 1 for r in rows})


FIGURES = {
    "fig3e_peak_rss_best": fig3e_peak_rss_best,
    "fig3f_cpu_wall_best": fig3f_cpu_wall_best,
    "fig2c_cost_vs_input": fig2c_cost_vs_input,
    "figS_cpu_scaling": figS_cpu_scaling,
    "figS_end_to_end": figS_end_to_end,
}

P.PLOTS.update(FIGURES)


def write_losses(fig_dir, results, dataset=""):
    """``<fig_dir>/losses.csv``: every loss row of the cost figures drawn, and one
    ``plot_placeholder`` row per cost figure that is its placeholder. ALWAYS written (a
    header alone when nothing was lost)."""
    rows = []
    for name, res in results.items():
        if name not in FIGURES:
            continue
        if res.get("placeholder"):
            rows.append(losses_mod.row(stage="paper", dataset=dataset, cause="plot_placeholder",
                                       effect="dropped", detail=f"{name}: input absent or all NA",
                                       source=f"figures/{name}"))
        rows += res.get("losses") or []
    path = Path(fig_dir) / "losses.csv"
    losses_mod.write(path, rows)
    return path
