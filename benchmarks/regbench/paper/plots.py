"""The paper's plots, drawn from the paper tables (``<out>/paper/*.csv``, Task 13-15).

``PLOTS`` maps a figure name to ``fn(paper_dir, fig_dir) -> {"files", "n", "whiskers"}``.
Every figure goes through ``plotstyle`` (the ``plot`` skill's canonical style, copied
unmodified): ``apply_style``, ``lock_palette`` (one colour per method family, the same in
every figure of ``fig_dir``), ``figure``, ``panel_labels`` and ``save`` (pdf, png, svg and
``<stem>_source_data.csv``).

**Placeholders (R22).** A figure whose input table is absent or empty is still written: a
page that names the table and the path it was expected at, returned with
``"placeholder": True``. A figure is never skipped silently.

**Boxes.** Median, IQR, whiskers at 1.5 x IQR, and every point drawn. A group of fewer
than ten points is drawn as points with a median bar and no box.

**Dice.** Read exactly as the tables read it: a ``dice_matched`` that is missing on a
non-excluded row counts as 0 (``tables._dice0``), an excluded row never enters, and a case
that has no matched pair on every compared row is left out (``tables.best_of``). Each tick
label gives n and how many cases were zeroed.

**VALIS stages.** ``initial``, ``rigid`` (depth 0), ``micro_rigid``, ``nonrigid`` and
``micro`` (depth 2): a stage a run does not have is a gap in the line, never a point.
"""
from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from . import plotstyle as ps  # noqa: E402
from . import schema, tables  # noqa: E402

WHISKERS = "1.5 x IQR"
MIN_BOX = 10  # fewer points than this: points with a median bar, never a box
FAMILIES = (
    "stare",
    "valis_mirage",
    "valis120",
    "deeperhistreg",
    "ashlar",
    "affine_ashlar",
    "palom",
)
LABELS = {
    "stare": "STARE",
    "valis_mirage": "VALIS (Mirage)",
    "valis120": "VALIS 1.2.0",
    "deeperhistreg": "DeeperHistReg",
    "ashlar": "ASHLAR",
    "affine_ashlar": "Affine ASHLAR",
    "palom": "PALOM",
}
STAGES = ("initial", "rigid", "micro_rigid", "nonrigid", "micro")
TIERS = ("high", "medium", "low")
EXPECTED = "expected"  # the collector's rows that hold no measurement yet
DRAFT_SUFFIX = "_draft"
DRAFT_COLOR = "#B03A2E"


# How a figure is drawn from a tables folder (``render_tables``): set only inside it.
#   stamp          "DRAFT — interim data (<sources>)": on every figure, ``_draft`` on its name
#   fallback       a folder whose derived table stands in for an EMPTY one of the folder
#   schema_only    refuse to read a CSV that is not a paper-schema table
#   drop_expected  leave out the collector's ``source = expected`` rows (no measurement)
_CTX = {}


# ── reading ──────────────────────────────────────────────────────────────────
def _read_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, newline="") as fh:
        rows = [{k: (None if v == "NA" else v) for k, v in r.items()} for r in csv.DictReader(fh)]
    if _CTX.get("drop_expected"):
        rows = _case_keyed([r for r in rows if r.get("source") != EXPECTED])
    return rows or None


def _case_keyed(rows):
    """A row that names no ``case_id`` but its patient and panel (an arm's row: the arms
    know no case ids) is keyed ``<patient>/<panel>``, so its cases stay apart."""
    for r in rows:
        if "case_id" in r and r["case_id"] is None and r.get("patient_id") and r.get("panel"):
            r["case_id"] = f"{r['patient_id']}/{r['panel']}"
    return rows


def _read(path):
    path = Path(path)
    if _CTX.get("schema_only") and path.stem not in schema.TABLES:
        raise ValueError(f"plots: {path} is not a paper-schema table (paper/schema.py)")
    rows = _read_file(path)
    fallback = _CTX.get("fallback")
    if rows is None and fallback is not None and (Path(fallback) / path.name).is_file():
        rows = _read_file(Path(fallback) / path.name)
    return rows


def _num(v):
    return tables._num(v)


def _fin(v):
    return math.isfinite(_num(v))


def _flag(v):
    return tables._flag(v)


def _label(family):
    return LABELS.get(family, family)


class _Table:
    """The plotted values, with the ``to_csv`` that ``plotstyle.save`` expects."""

    def __init__(self, cols, rows):
        self.cols, self.rows = list(cols), list(rows)

    def to_csv(self, path, index=False):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.cols)
            for r in self.rows:
                w.writerow(["NA" if r.get(c) is None else r.get(c) for c in self.cols])


def _style():
    ps.apply_style()


def _colors(fig_dir, extra=()):
    cats = list(FAMILIES) + [e for e in extra if e not in FAMILIES]
    return ps.lock_palette(cats, fig_dir)


def _save(fig, fig_dir, name, data):
    stamp = _CTX.get("stamp")
    if stamp:
        fig.suptitle(stamp, fontsize=ps.TICK_PT, color=DRAFT_COLOR, fontweight="bold")
        name = f"{name}{DRAFT_SUFFIX}"
    return [str(p) for p in ps.save(fig, Path(fig_dir) / name, data=data)]


def _result(files, n, **kw):
    return {"files": files, "n": n, "whiskers": WHISKERS, **kw}


def _placeholder(fig_dir, name, missing):
    """The page that stands in for a figure whose input is absent: names each missing
    table and the path it was expected at."""
    _style()
    fig, (ax,) = ps.figure(width="onehalf", aspect=0.5)
    ax.axis("off")
    lines = [f"{name}: not drawn, input missing"]
    for table, path in missing:
        lines.append(f"table: {table}")
        lines += textwrap.wrap(f"expected at: {path}", 70, break_long_words=True,
                               break_on_hyphens=False)
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", transform=ax.transAxes,
            fontsize=ps.TICK_PT)
    data = _Table(["table", "expected_path"], [{"table": t, "expected_path": str(p)} for t, p in missing])
    files = _save(fig, fig_dir, name, data)
    return _result(files, {}, placeholder=True)


def _need(paper_dir, **names):
    """``({key: rows}, missing)`` for the named tables (``key=filename``)."""
    got, missing = {}, []
    for key, fname in names.items():
        rows = _read(Path(paper_dir) / fname)
        if rows is None:
            missing.append((fname.removesuffix(".csv"), Path(paper_dir) / fname))
        else:
            got[key] = rows
    return got, missing


# ── per-case dice ────────────────────────────────────────────────────────────
def _outlines(per_pair, set_=tables.OUTLINES):
    # a cost-only row (stare_local4) is in no accuracy figure, even in a per_pair.csv
    # written before T31's fix C1 or collected from a draft
    return [r for r in per_pair if r.get("set") == set_
            and not _flag(r.get("excluded_from_summaries"))
            and not tables.is_cost_only(r.get("row"))]


def _cases(per_pair, row, set_=tables.OUTLINES):
    """``{case_id: record}`` of one result name: kept rows of one object set (the
    outlines unless said), dice read as the tables read it."""
    out = {}
    for r in _outlines(per_pair, set_):
        if r["row"] != row:
            continue
        out[r["case_id"]] = {
            "case_id": r["case_id"],
            "patient_id": r.get("patient_id"),
            "dice": tables._dice0(r),
            "zeroed": tables._no_pair(r),
            "disp": _num(r.get("displacement_um_p50")),
            "readable": _flag(r.get("displacement_readable")),
        }
    return out


def _drop_empty_cases(by_row):
    """``best_of``'s rule: a case that has no matched pair on every compared row is the
    case's, not a method's, and leaves the comparison."""
    ids = {k for cs in by_row.values() for k in cs}
    gone = {k for k in ids if all(cs[k]["zeroed"] for cs in by_row.values() if k in cs)}
    return {row: {k: v for k, v in cs.items() if k not in gone} for row, cs in by_row.items()}


def _dataset(per_pair):
    return next((r["dataset"] for r in per_pair if r.get("dataset")), tables.COHORT)


# ── drawing ──────────────────────────────────────────────────────────────────
def _draw_group(ax, pos, values, color, horizontal=False, solid=True, seed=0, hollow=False):
    """One group: box (n >= 10) or median bar (n < 10), every point drawn; ``hollow``
    draws the points and the box as outlines (T31 rule 3: a family's best-Dice row,
    "best (ceiling)", beside its filled primary row)."""
    v = np.asarray([x for x in values if math.isfinite(x)], float)
    if v.size == 0:
        return
    xy = (lambda a, b: (b, a)) if horizontal else (lambda a, b: (a, b))
    rng = np.random.default_rng(seed)
    jitter = pos + rng.uniform(-0.16, 0.16, v.size)
    alpha = 1.0 if solid else 0.45
    med = float(np.median(v))
    if v.size >= MIN_BOX:
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        lo = v[v >= q1 - 1.5 * iqr].min()
        hi = v[v <= q3 + 1.5 * iqr].max()
        w = 0.3
        if not hollow:
            ax.fill(*xy([pos - w, pos + w, pos + w, pos - w], [q1, q1, q3, q3]),
                    facecolor=color, alpha=0.25 if solid else 0.12, edgecolor="none", zorder=1)
        ax.plot(*xy([pos - w, pos + w, pos + w, pos - w, pos - w], [q1, q1, q3, q3, q1]),
                color=color, lw=0.6, alpha=alpha, zorder=2)
        ax.plot(*xy([pos, pos], [lo, q1]), color=color, lw=0.6, alpha=alpha, zorder=2)
        ax.plot(*xy([pos, pos], [q3, hi]), color=color, lw=0.6, alpha=alpha, zorder=2)
        ax.plot(*xy([pos - w, pos + w], [med, med]), color=ps.GREY, lw=1.0, zorder=4)
    else:
        ax.plot(*xy([pos - 0.3, pos + 0.3], [med, med]), color=ps.GREY, lw=1.2, zorder=4)
    if hollow:
        ax.scatter(*xy(jitter, v), s=6, facecolor="none", edgecolor=color,
                   alpha=0.7 * alpha + 0.2, linewidths=0.5, zorder=3)
    else:
        ax.scatter(*xy(jitter, v), s=4, color=color, alpha=0.7 * alpha + 0.2, linewidths=0,
                   zorder=3)


def _group_axes(ax, labels, horizontal=False, bold=()):
    pos = list(range(len(labels)))
    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set_ylim(len(labels) - 0.4, -0.6)
        for t, b in zip(ax.get_yticklabels(), [i in bold for i in pos]):
            if b:
                t.set_fontweight("bold")
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        for t, b in zip(ax.get_xticklabels(), [i in bold for i in pos]):
            if b:
                t.set_fontweight("bold")


def _tick(name, vals, zeroed=None, extra=""):
    t = f"{name}\nn={len(vals)}"
    if zeroed is not None:
        t += f", {zeroed} zero"
    return t + extra


def _distribution(fig_dir, name, groups, ylabel, value_key, ylim=None, log=False, cols=None,
                  width="double", aspect=0.6, zero=True, empty=()):
    """A one-panel box figure over ``groups`` = [(label, family, [records])]."""
    _style()
    colors = _colors(fig_dir)
    fig, (ax,) = ps.figure(width=width, aspect=aspect)
    labels, rows = [], []
    for i, (lab, fam, recs) in enumerate(groups):
        vals = [r[value_key] for r in recs]
        _draw_group(ax, i, vals, colors[fam], seed=i)
        z = sum(r.get("zeroed", False) for r in recs) if zero else None
        labels.append(_tick(lab, vals, z))
        for r in recs:
            rows.append({"family": fam, "row": r.get("row_name"), "case_id": r["case_id"],
                         "patient_id": r.get("patient_id"), value_key: r[value_key],
                         "zeroed": r.get("zeroed") if zero else None})
    _group_axes(ax, labels)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    cols = cols or ["family", "row", "case_id", "patient_id", value_key, "zeroed"]
    files = _save(fig, fig_dir, name, _Table(cols, rows))
    return files


# ── Fig. 3b ──────────────────────────────────────────────────────────────────
def _of_set(r, set_):
    """A method_best row of the object set ``set_``; a row with no ``set`` (a table
    written before T29) is the outlines'."""
    have = r.get("set")
    return have == set_ or (set_ == tables.OUTLINES and have in (None, "", schema.NA))


def _best_rows(best, set_=tables.OUTLINES):
    """``[(family, best_row or None)]`` of one object set (T29: method_best has a row per
    family per set; the outlines' unless said) in the table's order, plus every canonical
    family the table does not list (no row: n of 0)."""
    seen = []
    for r in best:
        if _of_set(r, set_):
            seen.append((r["family"], r.get("best_row")))
    have = {f for f, _ in seen}
    return seen + [(f, None) for f in FAMILIES if f not in have]


def _best_cases(per_pair, best, set_=tables.OUTLINES):
    picks = [(f, row) for f, row in _best_rows(best, set_) if row]
    by_row = {row: _cases(per_pair, row, set_) for _, row in picks}
    by_row = _drop_empty_cases(by_row)
    groups, n = [], {}
    for f, row in _best_rows(best, set_):
        cs = list(by_row.get(row, {}).values()) if row else []
        for c in cs:
            c["row_name"] = row
        n[f] = len(cs)
        if cs:
            groups.append((_label(f), f, cs, row))
    return groups, n


def fig3b_dice_best(paper_dir, fig_dir):
    got, missing = _need(paper_dir, pp="per_pair.csv", best="method_best.csv")
    if missing:
        return _placeholder(fig_dir, "fig3b_dice_best", missing)
    groups, n = _best_cases(got["pp"], got["best"])
    files = _distribution(fig_dir, "fig3b_dice_best", [g[:3] for g in groups],
                          "Matched-pair Dice (outlines, unitless)", "dice", ylim=(-0.03, 1.03))
    return _result(files, n)


def fig3b_dice_best_nuclei(paper_dir, fig_dir):
    """Fig. 3b's nuclei panel (T29): each family's best row on the bare nuclei (its
    ``method_best`` row of ``set = nuclei``, chosen on the nuclei), over the nuclei's
    matched-pair Dice; a placeholder when no nuclei row is there."""
    got, missing = _need(paper_dir, pp="per_pair.csv", best="method_best.csv")
    if not missing and not any(_of_set(r, tables.NUCLEI) and r.get("best_row")
                               for r in got["best"]):
        missing = [("method_best: no best row of set nuclei",
                    Path(paper_dir) / "method_best.csv")]
    if missing:
        return _placeholder(fig_dir, "fig3b_dice_best_nuclei", missing)
    groups, n = _best_cases(got["pp"], got["best"], tables.NUCLEI)
    files = _distribution(fig_dir, "fig3b_dice_best_nuclei", [g[:3] for g in groups],
                          "Matched-pair Dice (bare nuclei, unitless)", "dice",
                          ylim=(-0.03, 1.03))
    return _result(files, n)


def _terminal(per_pair):
    ds = _dataset(per_pair)
    try:
        return set(tables.terminal_rows(ds).values())
    except KeyError:
        return {r["row"] for r in per_pair
                if r.get("stage") != tables.INITIAL and not tables.is_cost_only(r["row"])}


def fig3b_dice_all(paper_dir, fig_dir):
    got, missing = _need(paper_dir, pp="per_pair.csv", best="method_best.csv")
    if missing:
        return _placeholder(fig_dir, "fig3b_dice_all", missing)
    pp = got["pp"]
    best_of_family = {f: row for f, row in _best_rows(got["best"]) if row}
    best_set = set(best_of_family.values())
    terminal = _terminal(pp)
    famof = {}
    for r in pp:
        if r.get("family"):
            famof[r["row"]] = r["family"]
    rows = [r for r in dict.fromkeys(r["row"] for r in _outlines(pp)) if r in terminal and r in famof]
    by_row = _drop_empty_cases({r: _cases(pp, r) for r in rows})
    order = [f for f, _ in _best_rows(got["best"])]
    order += [f for f in dict.fromkeys(famof.values()) if f not in order]
    _style()
    colors = _colors(fig_dir, extra=order)
    groups = []
    n = {f: 0 for f in order}
    for f in order:
        for row in rows:
            if famof[row] != f or not by_row.get(row):
                continue
            cs = list(by_row[row].values())
            groups.append((row, f, cs))
            n[row] = len(cs)
            n[f] += len(cs)
    height = min(ps.MAX_HEIGHT_MM, 14 + 4.2 * max(len(groups), 1))
    fig, (ax,) = ps.figure(width="double", height_mm=height)
    labels, data, bold = [], [], []
    for i, (row, f, cs) in enumerate(groups):
        hi = row in best_set
        _draw_group(ax, i, [c["dice"] for c in cs], colors[f], horizontal=True, solid=hi, seed=i)
        z = sum(c["zeroed"] for c in cs)
        labels.append(f"{row}  (n={len(cs)}, {z} zero)")
        if hi:
            bold.append(i)
        for c in cs:
            data.append({"family": f, "row": row, "case_id": c["case_id"],
                         "patient_id": c["patient_id"], "dice": c["dice"],
                         "zeroed": c["zeroed"], "best_of_family": hi})
    _group_axes(ax, labels, horizontal=True, bold=bold)
    ax.set_xlabel("Matched-pair Dice (outlines, unitless); bold: the family's best row")
    ax.set_xlim(-0.03, 1.03)
    handles = [plt.Line2D([], [], marker="s", ls="", color=colors[f], label=_label(f))
               for f in order if any(g[1] == f for g in groups)]
    if handles:
        fig.legend(handles=handles, loc="outside right center", title="Method")
    files = _save(fig, fig_dir, "fig3b_dice_all",
                  _Table(["family", "row", "case_id", "patient_id", "dice", "zeroed",
                          "best_of_family"], data))
    return _result(files, n)


# ── Fig. 3c ──────────────────────────────────────────────────────────────────
def fig3c_displacement_best(paper_dir, fig_dir):
    got, missing = _need(paper_dir, pp="per_pair.csv", best="method_best.csv")
    if missing:
        return _placeholder(fig_dir, "fig3c_displacement_best", missing)
    groups, _ = _best_cases(got["pp"], got["best"])
    out, n = [], {f: 0 for f, _ in _best_rows(got["best"])}
    for lab, f, cs, row in groups:
        rd = [c for c in cs if c["readable"] and math.isfinite(c["disp"])]
        n[f] = len(rd)
        if rd:
            out.append((lab, f, rd))
    files = _distribution(fig_dir, "fig3c_displacement_best", out,
                          "Median centroid displacement (µm)", "disp",
                          cols=["family", "row", "case_id", "patient_id", "disp"], zero=False)
    return _result(files, n)


# ── Fig. 3d ──────────────────────────────────────────────────────────────────
def fig3d_valis_stages(paper_dir, fig_dir):
    got, missing = _need(paper_dir, vs="valis_stages.csv")
    if missing:
        return _placeholder(fig_dir, "fig3d_valis_stages", missing)
    _style()
    colors = _colors(fig_dir)
    runs = {}
    for r in got["vs"]:
        # T29: the outlines on the common cases, and no excluded run's row of NA
        if tables.is_headline(r) and r.get("excluded_reason") in (None, "", schema.NA):
            runs.setdefault(r["row"], {})[r["stage"]] = r
    styles = {"high": "-", "medium": "--", "low": ":"}
    marks = {"0": "o", "1": "s", "2": "^"}
    fig, (ax,) = ps.figure(width="onehalf", aspect=0.75)
    data, n = [], {}
    for row, st in runs.items():
        first = next(iter(st.values()))
        ys = [_num(st[s]["dice_med"]) if s in st and st[s].get("stage_row") else math.nan
              for s in STAGES]
        xs = np.arange(len(STAGES))
        ax.plot(xs, ys, ls=styles.get(first["memory_mode"], "-"), color=colors["valis_mirage"],
                marker=marks.get(str(first["micro_reg"]), "o"), ms=3, lw=0.8,
                label=f"{first['memory_mode']}, depth {first['micro_reg']}")
        n[row] = max((int(_num(st[s]["n_pairs"])) for s in st if _fin(st[s].get("n_pairs"))), default=0)
        for s, y in zip(STAGES, ys):
            data.append({"row": row, "memory_mode": first["memory_mode"],
                         "micro_reg": first["micro_reg"], "stage": s, "dice_med": y})
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([s.replace("_", "\n") for s in STAGES])
    ax.set_ylabel("Median matched-pair Dice (unitless)")
    ax.set_xlabel("Registration stage (a missing stage is a gap)")
    fig.legend(*ax.get_legend_handles_labels(), title="VALIS tier, micro depth",
               loc="outside right center")
    files = _save(fig, fig_dir, "fig3d_valis_stages",
                  _Table(["row", "memory_mode", "micro_reg", "stage", "dice_med"], data))
    return _result(files, n)


def fig3d_stare_tiers(paper_dir, fig_dir):
    got, missing = _need(paper_dir, pp="per_pair.csv", st="stare_tiers.csv")
    if missing:
        return _placeholder(fig_dir, "fig3d_stare_tiers", missing)
    picks = {}
    for r in got["st"]:
        if (not tables.is_headline(r) or tables.is_cost_only(r.get("row"))
                or r.get("excluded_reason") not in (None, "", schema.NA)):
            continue  # a cost row is no tier of STARE's (T31 fix C1)
        if _num(r.get("stride")) == 64 and (r.get("sprep") or "none") == "none":
            picks[r["tier"]] = r["row"]
    by_row = _drop_empty_cases({row: _cases(got["pp"], row) for row in picks.values()})
    groups, n = [], {}
    for tier in TIERS:
        row = picks.get(tier)
        cs = list(by_row.get(row, {}).values()) if row else []
        for c in cs:
            c["row_name"] = row
        n[tier] = len(cs)
        if cs:
            groups.append((tier, "stare", cs))
    files = _distribution(fig_dir, "fig3d_stare_tiers", groups,
                          "Matched-pair Dice (outlines, unitless)", "dice", ylim=(-0.03, 1.03),
                          width="single", aspect=0.9)
    return _result(files, n)


# ── Fig. 3e, 3f, 2c, the CPU cross and end to end: paper/cost_plots.py (T31) ──────


# ── supplementary ────────────────────────────────────────────────────────────
def figS_pipeline_cost_fits(paper_dir, fig_dir):
    got, missing = _need(paper_dir, fits="pipeline_cost_fits.csv")
    if missing:
        return _placeholder(fig_dir, "figS_pipeline_cost_fits", missing)
    ok = [r for r in got["fits"] if r.get("target") == "peak_rss_gb" and _flag(r.get("fit_ok"))
          and _fin(r.get("slope"))]
    if not ok:
        return _placeholder(fig_dir, "figS_pipeline_cost_fits",
                            [("pipeline_cost_fits (no fit_ok row of peak_rss_gb)",
                              Path(paper_dir) / "pipeline_cost_fits.csv")])
    _style()
    backends = sorted({r["backend"] for r in ok})
    colors = _colors(fig_dir, extra=[b for b in backends if b not in FAMILIES])
    ok.sort(key=lambda r: (r["backend"], r["process"]))
    height = min(ps.MAX_HEIGHT_MM, 16 + 5.0 * len(ok))
    fig, axes = ps.figure(width="double", height_mm=height, nrows=1, ncols=2,
                          sharey=True)
    labels = [
        f"{r['process']}\n({r['backend']}, n={r['n']}"
        + (f", {int(_num(r['n_dropped']))} dropped" if _num(r.get("n_dropped")) > 0 else "")
        + ")"
        for r in ok
    ]
    data = []
    for ax, key, xlab in ((axes[0], "slope", "Slope: peak memory per input size (GB per GB)"),
                          (axes[1], "intercept", "Intercept: peak memory at no input (GB)")):
        for i, r in enumerate(ok):
            col = colors.get(r["backend"], ps.GREY)
            ax.scatter([_num(r[key])], [i], s=14, color=col, linewidths=0, zorder=3)
        ax.axvline(0, color=ps.LIGHT_GREY, lw=0.5, zorder=1)
        ax.set_xlabel(xlab)
        ax.set_yticks(range(len(ok)))
        ax.set_yticklabels(labels)
        ax.set_ylim(len(ok) - 0.4, -0.6)
    for r in ok:
        data.append({"backend": r["backend"], "process": r["process"], "n": r["n"],
                     "slope": _num(r["slope"]), "intercept": _num(r["intercept"]),
                     "r2": _num(r.get("r2"))})
    ps.panel_labels(axes)
    files = _save(fig, fig_dir, "figS_pipeline_cost_fits",
                  _Table(["backend", "process", "n", "slope", "intercept", "r2"], data))
    return _result(files, {f"{r['backend']}:{r['process']}": int(_num(r["n"])) for r in ok})


# ── public datasets ──────────────────────────────────────────────────────────
def _public_dirs(paper_dir):
    """The dataset folders under ``paper_dir`` that hold a ``per_pair.csv`` (the public
    nuclei datasets), or ``paper_dir`` itself when it is such a dataset's folder."""
    base = Path(paper_dir)
    subs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "per_pair.csv").is_file()) \
        if base.is_dir() else []
    if subs:
        return subs
    rows = _read(base / "per_pair.csv")
    if rows and _dataset(rows) != tables.COHORT:
        return [base]
    return []


def public_dice_best(paper_dir, fig_dir):
    dirs = _public_dirs(paper_dir)
    if not dirs:
        return _placeholder(fig_dir, "public_dice_best",
                            [("per_pair (a public nuclei dataset)",
                              Path(paper_dir) / "<dataset>" / "per_pair.csv"),
                             ("method_best", Path(paper_dir) / "<dataset>" / "method_best.csv")])
    _style()
    colors = _colors(fig_dir)
    fig, axes = ps.figure(width="double", nrows=1, ncols=len(dirs), aspect=0.9 if len(dirs) < 3 else 1.3,
                          sharey=True)
    n, data = {}, []
    for ax, d in zip(axes, dirs):
        pp, best = _read(d / "per_pair.csv"), _read(d / "method_best.csv")
        if pp is None or best is None:
            ax.axis("off")
            ax.text(0, 1, f"{d.name}: missing\n{d / 'method_best.csv'}", va="top",
                    fontsize=ps.TICK_PT)
            continue
        groups, nn = _best_cases(pp, best)
        labels = []
        for i, (lab, f, cs, row) in enumerate(groups):
            _draw_group(ax, i, [c["dice"] for c in cs], colors[f], seed=i)
            labels.append(_tick(lab, cs, sum(c["zeroed"] for c in cs)))
            for c in cs:
                data.append({"dataset": d.name, "family": f, "row": row, "case_id": c["case_id"],
                             "dice": c["dice"], "zeroed": c["zeroed"]})
        _group_axes(ax, labels)
        for t in ax.get_xticklabels():
            t.set_rotation(90)
        ax.set_title(d.name)
        ax.set_ylim(-0.03, 1.03)
        n.update({f"{d.name}:{f}": v for f, v in nn.items()})
    axes[0].set_ylabel("Matched-pair Dice (outlines, unitless)")
    ps.panel_labels(axes)
    files = _save(fig, fig_dir, "public_dice_best",
                  _Table(["dataset", "family", "row", "case_id", "dice", "zeroed"], data))
    return _result(files, n)


def public_landmarks_best(paper_dir, fig_dir):
    """Per landmark dataset, the best row of each family (lowest ``rtre_median_med``):
    the table has the median over pairs of the median relative TRE, not the pairs."""
    base = Path(paper_dir)
    rows = []
    paths = [base / "public_landmarks.csv"] + (
        sorted(base.glob("*/public_landmarks.csv")) if base.is_dir() else [])
    for p in paths:
        rows += _read(p) or []
    if not rows:
        return _placeholder(fig_dir, "public_landmarks_best",
                            [("public_landmarks", base / "public_landmarks.csv")])
    # every pair of each set (HyReCo `all_pairs`), not a subset's rows
    # and only a method's rows: `initial` (no registration) has no family
    rows = [r for r in rows if _fin(r.get("rtre_median_med")) and r.get("family")
            and r["family"] != "NA"
            and r.get("set", "all_pairs") in ("all_pairs", "", None)]
    datasets = sorted({r["dataset"] for r in rows})
    _style()
    colors = _colors(fig_dir, extra=sorted({r["family"] for r in rows if r.get("family")}))
    fig, axes = ps.figure(width="double", nrows=1, ncols=len(datasets), aspect=1.0, sharey=False)
    n, data = {}, []
    for ax, ds in zip(axes, datasets):
        best = {}
        for r in rows:
            if r["dataset"] != ds:
                continue
            f = r["family"]
            if f not in best or _num(r["rtre_median_med"]) < _num(best[f]["rtre_median_med"]):
                best[f] = r
        fams = sorted(best, key=lambda f: _num(best[f]["rtre_median_med"]))
        for i, f in enumerate(fams):
            r = best[f]
            ax.scatter([_num(r["rtre_median_med"])], [i], s=14, color=colors[f], linewidths=0)
            data.append({"dataset": ds, "family": f, "row": r["row"], "n_pairs": r.get("n_pairs"),
                         "rtre_median_med": _num(r["rtre_median_med"])})
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([f"{_label(f)}\nn={best[f].get('n_pairs')} pairs" for f in fams])
        ax.set_ylim(len(fams) - 0.4, -0.6)
        ax.set_xlim(left=0)
        ax.set_xlabel("Median relative TRE over pairs (fraction of image diagonal)")
        ax.set_title(ds)
        n[ds] = len(fams)
    ps.panel_labels(axes)
    files = _save(fig, fig_dir, "public_landmarks_best",
                  _Table(["dataset", "family", "row", "n_pairs", "rtre_median_med"], data))
    return _result(files, n)


PLOTS = {
    "fig3b_dice_best": fig3b_dice_best,
    "fig3b_dice_best_nuclei": fig3b_dice_best_nuclei,
    "fig3b_dice_all": fig3b_dice_all,
    "fig3c_displacement_best": fig3c_displacement_best,
    "fig3d_valis_stages": fig3d_valis_stages,
    "fig3d_stare_tiers": fig3d_stare_tiers,
    "fig3e_peak_rss_best": None,  # cost_plots.FIGURES, filled in below
    "fig3f_cpu_wall_best": None,
    "fig2c_cost_vs_input": None,
    "figS_cpu_scaling": None,
    "figS_end_to_end": None,
    "figS_pipeline_cost_fits": figS_pipeline_cost_fits,
    "public_dice_best": public_dice_best,
    "public_landmarks_best": public_landmarks_best,
}


# ── all-NA inputs (R22) ──────────────────────────────────────────────────────
# Per figure, the tables it reads and the columns that carry what it draws. A figure whose
# table is absent, or whose columns are NA in every row of it, is the placeholder naming
# them (never an empty axis).
NEEDS = {
    "fig3b_dice_best": {"per_pair.csv": ("dice_matched",), "method_best.csv": ("best_row",)},
    "fig3b_dice_best_nuclei": {"per_pair.csv": ("dice_matched",),
                               "method_best.csv": ("best_row",)},
    "fig3b_dice_all": {"per_pair.csv": ("dice_matched",), "method_best.csv": ("best_row",)},
    "fig3c_displacement_best": {"per_pair.csv": ("displacement_um_p50",),
                                "method_best.csv": ("best_row",)},
    "fig3d_valis_stages": {"valis_stages.csv": ("dice_med",)},
    "fig3d_stare_tiers": {"per_pair.csv": ("dice_matched",), "stare_tiers.csv": ("row",)},
    "fig3e_peak_rss_best": {"method_best.csv": ("best_row",), "cost_points.csv": ("peak_rss_gb",)},
    "fig3f_cpu_wall_best": {"method_best.csv": ("best_row",),
                            "cost_points.csv": ("cpu_h_used", "wall_s")},
    "fig2c_cost_vs_input": {"method_best.csv": ("best_row",), "cost_points.csv": ("input_gpx",),
                            "cost_fits.csv": ("slope",)},
    "figS_cpu_scaling": {"cpu_scaling.csv": ("wall_s",)},
    "figS_end_to_end": {"end_to_end.csv": ("estimate_s_med",)},
    "figS_pipeline_cost_fits": {"pipeline_cost_fits.csv": ("slope",)},
}


def _not_drawable(paper_dir, name):
    """``[(what, path)]``: each table of ``NEEDS[name]`` that is absent, or whose columns
    are all NA; ``[]`` when the figure can be drawn."""
    missing = []
    for fname, cols in NEEDS.get(name, {}).items():
        path = Path(paper_dir) / fname
        rows = _read(path)
        table = fname.removesuffix(".csv")
        if rows is None:
            missing.append((table, path))
        elif all(r.get(c) is None for r in rows for c in cols):
            missing.append((f"{table}: column(s) {', '.join(cols)} NA in all {len(rows)} row(s)",
                            path))
    return missing


def draw(name, paper_dir, fig_dir):
    """One figure of ``PLOTS``: its placeholder when its inputs are absent or all NA."""
    missing = _not_drawable(paper_dir, name)
    if missing:
        return _placeholder(fig_dir, name, missing)
    return PLOTS[name](paper_dir, fig_dir)


def _only(only, known):
    names = [s for s in (only or "").split(",") if s]
    unknown = [s for s in names if s not in known]
    if unknown:
        raise SystemExit(f"plots: no figure {unknown}; known: {sorted(known)}")
    return names


def main(a):
    base = Path(a.out) / "paper"
    paper_dir = base if a.dataset == tables.COHORT else base / a.dataset
    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    only = _only(a.only, PLOTS)
    got = {}
    for name in only or PLOTS:
        res = got[name] = draw(name, paper_dir, fig_dir)
        tag = "placeholder" if res.get("placeholder") else "drawn"
        print(f"{name}\t{tag}\t{res['files'][0]}")
    print(f"losses\t{cost_plots.write_losses(paper_dir / 'losses.csv', got, a.dataset)}")
    return 0


# ── drawing from a tables folder: drafts (Task 18b) ──────────────────────────
# ``regbench plots --tables DIR --to DIR [--draft]``. Every figure is drawn from the
# paper-schema CSVs of DIR and NOTHING else (``_CTX["schema_only"]``): a collection
# (``collect.py``) or a ``paper/`` folder. The collector's ``expected`` rows hold no
# measurement and are left out. A table the paper DERIVES from ``per_pair``
# (``DERIVABLE``: method_best, valis_stages, stare_tiers) that has no row in DIR is derived
# here by the paper's own rule (``tables.best_of`` / ``valis_stages`` / ``stare_tiers``,
# without cost) and written to ``<to>/derived/`` with ``source = derived:<DIR>``: what the
# figure drew is on disk. ``--draft`` (implied by the collector's README.txt) stamps
# ``DRAFT — interim data (<source kinds>)`` on every figure and adds ``_draft`` to its name.
# The overlay zooms need a results folder's images (``--results``, ``--cases``); without
# one, or where they were removed (``regbench slim``), each is the placeholder naming them.
DERIVABLE = ("method_best", "valis_stages", "stare_tiers")


def _typed(rows):
    out = []
    for r in rows:
        t = dict(r)
        for k in tables.NUMERIC:
            if k in t:
                t[k] = tables._num(t[k])
        for k in tables.FLAGS:
            if k in t:
                t[k] = tables._flag(t[k])
        out.append(t)
    return out


def _derive(tables_dir, to, dataset):
    """Write ``<to>/<table>.csv`` for each ``DERIVABLE`` table that has no row in
    ``tables_dir``; ``{table: rows written}``."""
    pp = _read_file(Path(tables_dir) / "per_pair.csv")
    if pp is None:
        return {}
    pp = _typed(_case_keyed([r for r in pp if r.get("source") != EXPECTED]))
    pp = tables.accuracy_pairs([r for r in pp if r.get("dataset") == dataset])
    if not pp:
        return {}
    source = f"derived:{tables_dir}"
    made = {}
    for name in DERIVABLE:
        if _read_file(Path(tables_dir) / f"{name}.csv") is not None:
            continue
        if name == "method_best":
            rows = [tables.best_of(pp, f, dataset, {}) for f in tables._families(dataset)]
        elif name == "valis_stages":
            rows = tables.valis_stages(pp, dataset)
        else:
            rows = tables.stare_tiers(pp, [], dataset)
        rows = [{**r, "source": source} for r in rows]
        Path(to).mkdir(parents=True, exist_ok=True)
        schema.write(Path(to) / f"{name}.csv", name, rows)
        made[name] = len(rows)
    if made:
        (Path(to) / "README.txt").write_text(
            f"Derived by `regbench plots --tables {tables_dir}` from its per_pair.csv (the "
            "collector's `expected` rows left out), by the paper's own rules "
            "(benchmarks/regbench/paper/tables.py: best_of, valis_stages, stare_tiers), "
            "without cost (no tie-break by CPU time): these tables had no row there.\n"
            + "".join(f"  {k}: {n} row(s)\n" for k, n in made.items()))
    return made


def sources(tables_dir):
    """The source kinds of the rows of ``tables_dir`` (``arms``, ``sweep``,
    ``regbench-interim``, ``regbench`` ...), ``expected`` left out, in order."""
    kinds = []
    for name in schema.TABLES:
        for r in _read_file(Path(tables_dir) / f"{name}.csv") or []:
            k = (r.get("source") or "").split(":", 1)[0]
            if k and k != EXPECTED and k not in kinds:
                kinds.append(k)
    return sorted(kinds)


def _overlay_placeholders(fig_dir, results, extra=()):
    from . import overlays

    where = Path(results) if results else Path("<results>")
    missing = [*extra,
               ("warped.npz of each best row (the rigid and final panels' maps)",
                where / "<row>" / "<dataset>" / "<case>" / "warped.npz"),
               ("the image each best row wrote (run.json native.files)",
                where / "<row>" / "<dataset>" / "<case>" / "native")]
    if not results:
        missing.insert(0, ("a results folder and its cases (plots --results, --cases): "
                           "none given", where))
    return {name: _placeholder(fig_dir, name, missing) for name in overlays.OVERLAYS}


def _overlays(tables_dir, to, results, cases, dataset, derived):
    from .. import slimmed
    from . import overlays

    if not results or not cases:
        return _overlay_placeholders(to, results)
    extra = []
    if slimmed.said(results):
        extra.append((f"removed by `regbench slim` ({slimmed.said(results)})",
                      Path(results) / slimmed.NAME))
    best = Path(tables_dir) / "method_best.csv"
    if _read(best) is None and (Path(derived) / "method_best.csv").is_file():
        best = Path(derived) / "method_best.csv"
    return overlays.build(cases, results, dataset=dataset, best_path=best, fig_dir=to,
                          rois_path=Path(to) / "overlay_rois.csv", extra_missing=extra)


def render_tables(tables_dir, to, draft=None, only=None, results=None, cases=None,
                  dataset=tables.COHORT):
    """Every figure of ``PLOTS`` and ``overlays.OVERLAYS`` from the paper-schema CSVs of
    ``tables_dir`` into ``to``; ``{figure: result}`` as ``PLOTS``."""
    from . import overlays, slim

    tables_dir, to = Path(tables_dir), Path(to)
    if not tables_dir.is_dir():
        raise SystemExit(f"plots --tables: {tables_dir} is not a folder")
    t, o = tables_dir.resolve(), to.resolve()
    if o == t or t in o.parents:
        raise SystemExit(f"plots --tables: --to {to} lies in the tables folder it reads")
    draft = bool(draft) or slim.is_collection(tables_dir)
    stamp = f"DRAFT — interim data ({', '.join(sources(tables_dir)) or 'no source'})" \
        if draft else None
    names = only if isinstance(only, (list, tuple)) else _only(only, [*PLOTS, *overlays.OVERLAYS])
    to.mkdir(parents=True, exist_ok=True)
    derived = to / "derived"
    _derive(tables_dir, derived, dataset)
    got = {}
    saved = dict(_CTX)
    _CTX.clear()
    _CTX.update(stamp=stamp, fallback=derived, schema_only=True, drop_expected=True)
    try:
        for name in PLOTS:
            if not names or name in names:
                got[name] = draw(name, tables_dir, to)
        if not names or set(names) & set(overlays.OVERLAYS):
            ov = _overlays(tables_dir, to, results, cases, dataset, derived)
            got.update({k: v for k, v in ov.items() if not names or k in names})
    finally:
        _CTX.clear()
        _CTX.update(saved)
    cost_plots.write_losses(to / "losses.csv", got, dataset)
    return got


def main_tables(a):
    if not a.to:
        raise SystemExit("plots --tables: --to <folder> is required")
    got = render_tables(a.tables, a.to, a.draft, a.only, a.results, a.cases, a.dataset)
    for name, res in got.items():
        tag = "placeholder" if res.get("placeholder") else "drawn"
        print(f"{name}\t{tag}\t{res['files'][0]}")
    return 0


# the cost figures live in their own module (T31): it reads this one's helpers and
# fills its entries of PLOTS when it is imported (either module may be imported first)
from . import cost_plots  # noqa: E402
