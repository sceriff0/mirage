"""ARCHIVED figure functions of benchmarks/regbench/paper (user ruling 2026-11-06).

The 261105 manuscripts show none of these figures, so they were taken out of the active
dispatch (``paper/plots.py``'s ``PLOTS`` and ``NEEDS``, ``paper/cost_plots.py``'s
``FIGURES``) and kept here verbatim rather than deleted. This file is NOT imported, run,
linted or tested (benchmarks/archive/README.md): the names it uses (``_need``,
``_placeholder``, ``_save``, ``ps``, ``P`` ...) are those of the module each function came
from. The code as it ran is at the tag archive/integrate-benchmarking-pre-prune-2026-11-06.
"""

# ── from benchmarks/regbench/paper/plots.py ─────────────────────────────────────────────

# ── Fig. 3d ──────────────────────────────────────────────────────────────────
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


def figS_synth_cross_scaling(paper_dir, fig_dir):
    """STARE's tier x stride cross and the three VALIS runs on the same axes: each run's
    fitted exponent (``synthetic_scaling``: the quantity grows as gigapixels^slope) with
    its 95% interval, one marker per quantity; a run without a fit is named, not drawn."""
    from .. import variants

    name = "figS_synth_cross_scaling"
    got, missing = _synth_need(paper_dir, "synthetic_scaling")
    if missing:
        return _placeholder(fig_dir, name, missing)
    scaling = [r for r in got["synthetic_scaling"] if not r.get("excluded_reason")]
    order = [r for r in variants.SYNTHETIC_RUNS]
    finals = tables.terminal_rows("synthetic")
    rows = [finals.get(r, r) for r in order]
    rows += sorted({r["row"] for r in scaling} - set(rows))
    _style()
    colors = _colors(fig_dir, extra=sorted({_run_style(r)[0] for r in rows}))
    fig, (ax,) = ps.figure(width="double", aspect=0.45)
    data, n, labels = [], {}, []
    keys = [k for k, _ in SYNTH_QUANTITIES]
    for i, row in enumerate(rows):
        fam, _, _, label = _run_style(row)
        drawn = 0
        for j, key in enumerate(keys):
            f = next((r for r in scaling if r["row"] == row and r["quantity"] == key), None)
            if f is None or not _flag(f.get("fit_ok")) or not _fin(f.get("slope")):
                continue
            x = i + (j - 1.5) * 0.15
            y = _num(f["slope"])
            lo, hi = _num(f.get("slope_ci_low")), _num(f.get("slope_ci_high"))
            if math.isfinite(lo) and math.isfinite(hi):
                ax.plot([x, x], [lo, hi], color=colors[fam], lw=0.7)
            ax.scatter([x], [y], s=14, marker=QUANTITY_MARKER[key], color=colors[fam],
                       label=key if i == 0 or key not in n else None)
            n[key] = n.get(key, 0) + 1
            drawn += 1
            data.append({"row": row, "quantity": key, "slope": y, "slope_ci_low": f.get(
                "slope_ci_low"), "slope_ci_high": f.get("slope_ci_high"),
                "n_sizes_used": f.get("n_sizes_used")})
        labels.append(label if drawn else f"{label}\n(no fit)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90)
    ax.axhline(1.0, color="0.6", lw=0.6, ls="--")
    ax.set_ylabel("Fitted exponent on gigapixels (95% interval)")
    handles = [matplotlib.lines.Line2D([], [], ls="", marker=QUANTITY_MARKER[k], color="0.3",
                                       label=lab) for k, lab in SYNTH_QUANTITIES]
    ax.legend(handles=handles, fontsize=ps.TICK_PT - 1)
    ps.panel_labels([ax])
    _synth_caveat(fig, scaling)
    files = _save(fig, fig_dir, name, _Table(
        ["row", "quantity", "slope", "slope_ci_low", "slope_ci_high", "n_sizes_used"], data))
    return _result(files, n)


def figS_synth_vs_cohort(paper_dir, fig_dir):
    """The synthetic fit at the cohort's median and largest patient against the cohort's
    measured cost of the same row, log-log, y = x drawn; a hollow marker is a fit used
    outside the sizes it was made on (``extrapolated``)."""
    name = "figS_synth_vs_cohort"
    got, missing = _synth_need(paper_dir, "synthetic_extrapolation")
    if missing:
        return _placeholder(fig_dir, name, missing)
    rows = [r for r in got["synthetic_extrapolation"] if not r.get("excluded_reason")
            and _fin(r.get("fitted")) and _fin(r.get("measured_cohort"))
            and _num(r["fitted"]) > 0 and _num(r["measured_cohort"]) > 0]
    if not rows:
        return _placeholder(fig_dir, name, [(
            "synthetic_extrapolation: no row with both a fitted and a measured cohort value "
            "(a fit of >= 3 sizes, and the cohort's cost tables)",
            _synth_path(paper_dir, f"{SYNTH}/synthetic_extrapolation.csv"))])
    _style()
    colors = _colors(fig_dir, extra=sorted({_run_style(r["row"])[0] for r in rows}))
    fig, axes = ps.figure(width="double", nrows=2, ncols=2, aspect=0.95)
    axes = np.asarray(axes).reshape(-1)
    data, n = [], {}
    for ax, (key, ylab) in zip(axes, SYNTH_QUANTITIES):
        mine = [r for r in rows if r["quantity"] == key]
        vals = []
        for r in mine:
            x, y = _num(r["measured_cohort"]), _num(r["fitted"])
            hollow = _flag(r.get("extrapolated"))
            ax.scatter([x], [y], s=16, marker="o" if r["at"] == "cohort_median" else "^",
                       facecolor="none" if hollow else colors[_run_style(r["row"])[0]],
                       edgecolor=colors[_run_style(r["row"])[0]], linewidths=0.7,
                       label=f"{_run_style(r['row'])[3]} ({r['at'].replace('cohort_', '')})")
            vals += [x, y]
            data.append({"quantity": key, "row": r["row"], "at": r["at"], "measured": x,
                         "fitted": y, "extrapolated": hollow, "at_gpx": r.get("at_gpx")})
        if vals:
            lo, hi = min(vals) / 1.5, max(vals) * 1.5
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.6, ls="--", label="y = x")
            ax.legend(fontsize=ps.TICK_PT - 1)
        n[key] = len(mine)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Cohort, measured: {ylab} (log10)")
        ax.set_ylabel(f"Synthetic fit: {ylab} (log10)")
    ps.panel_labels(list(axes))
    fig.text(0.0, 0.0, "circle: the cohort's median patient; triangle: its largest; hollow: "
             "extrapolated outside the synthetic sizes", fontsize=ps.TICK_PT - 1)
    files = _save(fig, fig_dir, name, _Table(
        ["quantity", "row", "at", "measured", "fitted", "extrapolated", "at_gpx"], data))
    return _result(files, n)


# ── from benchmarks/regbench/paper/cost_plots.py ────────────────────────────────────────

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
    # rule 4 (T31 fix M6): every axis names its clock (a table before I5: wall_clock)
    clocks = ", ".join(sorted({r.get("clock") or r.get("wall_clock") for r in got["cs"]}
                              - {None, "", "NA"})) or "not recorded"
    for ax, key, ylab in ((axes[0], "speedup_vs_4", f"Speed-up against 4 CPUs\n(wall on {clocks})"),
                          (axes[1], "cpu_h_used", f"CPU-hours per patient\n(clock: {clocks})"),
                          (axes[2], "peak_rss_gb", f"Peak memory (GB) per patient\n(clock: {clocks})")):
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
