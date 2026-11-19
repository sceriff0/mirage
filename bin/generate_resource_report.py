#!/usr/bin/env python3
"""
generate_resource_report.py
Self-contained HTML computational-resources report for a MIRAGE run.
Joins the aggregated input-size log with Nextflow's trace.txt. Stdlib only.

Its only caller is main.nf's `workflow.onComplete` (generateResourceReport), not a
process, so a grep of modules/ and conf/ will not find it. It runs on the launching
machine's interpreter outside every container, hence stdlib only and a plain
`print()` for the result line.
"""

import argparse
import csv
import html
import math
import re
from datetime import datetime, timezone
from pathlib import Path

_UNIT = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def parse_bytes(s):
    """Parse a Nextflow byte string ('3.2 GB', '512 MB', '-') to float bytes."""
    if s is None:
        return None
    s = s.strip()
    # Only "-" (Nextflow's not-run/cached sentinel) and empty are missing. A bare "0"
    # is a real zero-byte reading and falls through to 0.0, as in parse_duration.
    if not s or s == "-":
        return None
    m = re.match(r"^([\d.]+)\s*([KMGT]?B)$", s, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) * _UNIT[m.group(2).upper()], 1)
    try:
        return float(s)
    except ValueError:
        return None


def parse_duration(s):
    """Parse a duration ('12m 4s', '2h 1m', '1.5s', '-') to float seconds."""
    if s is None:
        return None
    s = s.strip()
    if not s or s == "-":
        return None
    total = 0.0
    found = False
    for value, unit in re.findall(r"([\d.]+)\s*(ms|s|m|h|d)", s):
        found = True
        v = float(value)
        total += {"ms": v / 1000, "s": v, "m": v * 60, "h": v * 3600, "d": v * 86400}[
            unit
        ]
    if found:
        return total
    try:
        return float(s)
    except ValueError:
        return None


def parse_percent(s):
    """Parse a percentage string ('142.3%', '-') to float.

    Rejects a value that parses but is not finite (``'nan%'``, ``'inf%'``), which
    ``float()`` accepts without error.
    """
    if s is None:
        return None
    s = s.strip().rstrip("%")
    if not s or s == "-":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _int_or_none(s):
    """Parse an integer trace field ('8', '-') to int or None."""
    if s is None:
        return None
    s = s.strip()
    if not s or s == "-":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_trace(path):
    """Parse trace.txt (TSV) into a list of per-task dicts with normalized fields."""
    rows = []
    if not path or not Path(path).exists():
        return rows
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            rows.append(
                {
                    "process": (r.get("process") or "").strip(),
                    "tag": (r.get("tag") or "").strip(),
                    "name": (r.get("name") or "").strip(),
                    "status": (r.get("status") or "").strip(),
                    "exit": (r.get("exit") or "").strip(),
                    "realtime_s": parse_duration(r.get("realtime")),
                    "duration_s": parse_duration(r.get("duration")),
                    "cpu_pct": parse_percent(r.get("%cpu")),
                    # The request, not the usage: needed to show over-provisioning.
                    "memory_b": parse_bytes(r.get("memory")),
                    "cpus": _int_or_none(r.get("cpus")),
                    "peak_rss_b": parse_bytes(r.get("peak_rss")),
                    "peak_vmem_b": parse_bytes(r.get("peak_vmem")),
                    "rchar_b": parse_bytes(r.get("rchar")),
                    "wchar_b": parse_bytes(r.get("wchar")),
                }
            )
    return rows


# The columns AGGREGATE_SIZE_LOGS writes, and the only header this reader accepts.
# Must equal lib/ProcessEnvelope.groovy's SIZE_LOG_COLUMNS
# (tests/test_size_log_schema_has_one_owner.py).
SIZE_LOG_COLUMNS = ("process", "sample_id", "filename", "bytes")


def parse_size_log(path):
    """Sum input bytes per (process, sample_id) from input_sizes.csv."""
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and tuple(reader.fieldnames) != SIZE_LOG_COLUMNS:
            raise ValueError(
                f"{path}: unexpected size-log header {tuple(reader.fieldnames)}; "
                f"expected {SIZE_LOG_COLUMNS}"
            )
        for row in reader:
            proc = (row.get("process") or "").strip()
            sample = (row.get("sample_id") or "").strip()
            # Stub runs write the real process name with 0 bytes, so no special case.
            if not proc:
                continue
            try:
                b = int(float(row.get("bytes") or 0))
            except ValueError:
                b = 0
            out[(proc, sample)] = out.get((proc, sample), 0) + b
    return out


def _maxf(values):
    """Max of the non-None floats, or None."""
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _sumf(values):
    """Sum of the non-None floats (0.0 if all None)."""
    return sum(v for v in values if v is not None)


def _is_failure(row):
    """A non-zero exit. `-` (Nextflow's not-run/cached sentinel) and `` (missing)
    are not failures."""
    return row.get("exit") not in ("0", "", "-", None)


_GIB = 1024**3


def _reserved_gb_h(row):
    """Reserved memory-hours for one task: GB requested x hours held.

    Falls back to observed ``peak_rss`` when no request was recorded. Returns 0.0 when
    neither is available.
    """
    rt = row.get("realtime_s")
    if not rt:
        return 0.0
    mem = row.get("memory_b")
    if mem is None:
        mem = row.get("peak_rss_b")
    if mem is None:
        return 0.0
    return (mem / _GIB) * (rt / 3600.0)


def rollup_by_process(trace_rows):
    """Aggregate trace rows into per-process resource summaries."""
    groups = {}
    for r in trace_rows:
        groups.setdefault(r["process"], []).append(r)
    out = []
    for proc, rows in sorted(groups.items()):
        rts = [r.get("realtime_s") for r in rows]
        realtime_total = _sumf(rts)
        n = len(rows)
        out.append(
            {
                "process": proc,
                "n_tasks": n,
                "realtime_total_s": realtime_total,
                "realtime_mean_s": round(realtime_total / n, 1) if n else 0.0,
                "cpu_max_pct": _maxf([r.get("cpu_pct") for r in rows]),
                "peak_rss_max_b": _maxf([r.get("peak_rss_b") for r in rows]),
                "peak_vmem_max_b": _maxf([r.get("peak_vmem_b") for r in rows]),
                "rchar_total_b": _sumf([r.get("rchar_b") for r in rows]),
                "wchar_total_b": _sumf([r.get("wchar_b") for r in rows]),
                "n_failed": sum(1 for r in rows if _is_failure(r)),
                "mem_req_max_b": _maxf([r.get("memory_b") for r in rows]),
                "failed_realtime_s": _sumf(
                    [r.get("realtime_s") for r in rows if _is_failure(r)]
                ),
                "failed_gb_h": sum(_reserved_gb_h(r) for r in rows if _is_failure(r)),
            }
        )
    return out


def join_size(trace_rows, size_map):
    """Attach input_bytes to each trace row via (process, tag), with prefix fallback."""
    # Pre-index sample ids by process for fallback matching.
    by_proc = {}
    for (proc, sample), b in size_map.items():
        by_proc.setdefault(proc, []).append((sample, b))
    joined = []
    for r in trace_rows:
        proc, tag = r["process"], r.get("tag", "")
        input_bytes = size_map.get((proc, tag))
        if input_bytes is None:
            # Fallback: a size sample whose id is a prefix of the trace tag
            # (tag is meta.id = "<patient>_<stem>"; sample is patient). Boundary-match on "_"
            # so "P1" does not match "P10_slide".
            for sample, b in by_proc.get(proc, []):
                if sample and (tag == sample or tag.startswith(sample + "_")):
                    input_bytes = b
                    break
        joined.append({**r, "input_bytes": input_bytes})
    return joined


def fmt_bytes(b):
    """Format a byte count for display, e.g. ``3.2 GB``.

    Parameters
    ----------
    b : float or None
        Bytes, or ``None`` for a missing reading.

    Returns
    -------
    str
        ``'N/A'`` when ``b`` is ``None``; ``'n/a'`` when ``b`` is not finite;
        otherwise one decimal place in the largest fitting unit, capped at TB.
    """
    if b is None:
        return "N/A"
    b = float(b)
    if not math.isfinite(b):
        return "n/a"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024


def fmt_secs(s):
    """Format a duration in seconds as ``1h 2m 3s``, dropping empty leading units.

    Parameters
    ----------
    s : float or None
        Seconds, or ``None`` for a missing reading.

    Returns
    -------
    str
        ``'N/A'`` when ``s`` is ``None``; ``'n/a'`` when ``s`` is not finite
        (``int()`` would raise on it).
    """
    if s is None:
        return "N/A"
    if not math.isfinite(s):
        return "n/a"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


# ---------------------------------------------------------------------------
# Inline SVG panels
#
# Hand-rolled and stdlib-only, in the shape of bin/generate_qc_report.py's SVG
# helpers. Not shared with it: this script runs on the head node outside every
# container and must import with the standard library alone.
# ---------------------------------------------------------------------------

_PANEL_W = 900  # viewBox width; the <svg> itself scales to its container
_ROW_H = 22  # one ranked bar per row
_LABEL_W = 260  # left gutter for process names
_VALUE_W = 150  # right gutter for the formatted value


def short_process(name):
    """`MIRAGE:PREPROCESSING:CONVERT_IMAGE` -> `CONVERT_IMAGE`.

    Bar labels drop the shared prefix; the full name stays in each bar's <title>
    tooltip.
    """
    return (name or "").rsplit(":", 1)[-1]


def _svg(width, height, body, title):
    """The one <svg> wrapper every panel here uses.

    A viewBox plus `width:100%` scales the panel to the report column, and
    `role`/`aria-label` name it for screen readers.
    """
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px;display:block" '
        f'role="img" aria-label="{_esc(title)}">{body}</svg>'
    )


def _finite_or_none(v):
    """A finite float, or ``None``: collapses ``+-inf``/``nan`` into "missing".

    Every panel already treats ``None`` as missing (`is not None`, `or 0.0`), so
    folding non-finite readings into it keeps them out of ``fmt_secs`` and out of SVG
    coordinates and labels.
    """
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v if math.isfinite(v) else None


def _positive_finite(v):
    """True iff `v` is a finite number strictly greater than zero.

    The only values `size_vs_runtime_svg`'s log-log axes can plot.
    """
    fv = _finite_or_none(v)
    return fv is not None and fv > 0


def walltime_bars_svg(roll, top=15):
    """Per-process wall-time contribution, ranked, longest first.

    A process with a non-finite `realtime_total_s` is dropped. Each bar's percentage
    is its share of the run's total wall-time; bar width scales to the longest bar
    shown (`vmax`).
    """
    roll = [
        r
        for r in roll
        if r.get("realtime_total_s") is None
        or _finite_or_none(r.get("realtime_total_s")) is not None
    ]
    total_all = _sumf([r.get("realtime_total_s") for r in roll]) or 1.0
    rows = sorted(roll, key=lambda r: -(r.get("realtime_total_s") or 0.0))[:top]
    if not rows:
        return ""
    vmax = max((r.get("realtime_total_s") or 0.0) for r in rows) or 1.0
    bar_w = _PANEL_W - _LABEL_W - _VALUE_W
    parts = []
    for i, r in enumerate(rows):
        y = i * _ROW_H
        v = r.get("realtime_total_s") or 0.0
        w = max(1.0, bar_w * v / vmax)
        pct = 100.0 * v / total_all
        parts.append(
            f"<text x='0' y='{y + 15}' font-size='12' fill='#333'>"
            f"{_esc(short_process(r['process']))}</text>"
            f"<rect class='bar' x='{_LABEL_W}' y='{y + 4}' width='{w:.1f}' "
            f"height='{_ROW_H - 8}' fill='#2c3e50'>"
            f"<title>{_esc(r['process'])}: {fmt_secs(v)} over "
            f"{r.get('n_tasks', 0)} task(s)</title></rect>"
            f"<text x='{_LABEL_W + bar_w + 8}' y='{y + 15}' font-size='12' "
            f"fill='#666'>{_esc(fmt_secs(v))} ({pct:.0f}%)</text>"
        )
    caption_y = len(rows) * _ROW_H + 14
    parts.append(
        f"<text x='{_LABEL_W}' y='{caption_y}' font-size='10' fill='#999'>"
        "% is each process's share of the run's total wall-time</text>"
    )
    return _svg(
        _PANEL_W,
        len(rows) * _ROW_H + 18,
        "".join(parts),
        "Per-process wall-time contribution, ranked, as % of the run's total wall-time",
    )


def memory_headroom_svg(roll, top=15):
    """Requested memory vs observed peak RSS, per process, ranked by waste.

    Two bars per process on a shared scale: a light track for the request and a dark
    overlay for the observed peak. A peak that overruns its request is drawn in red,
    not clipped. A process with a non-finite request or peak is skipped.
    """
    rows = [
        r
        for r in roll
        # Truthiness, not `is not None`: a 0.0 request would divide by zero below.
        if _finite_or_none(r.get("mem_req_max_b"))
        and _finite_or_none(r.get("peak_rss_max_b")) is not None
    ]
    rows.sort(key=lambda r: -(r["mem_req_max_b"] - r["peak_rss_max_b"]))
    rows = rows[:top]
    if not rows:
        return ""
    vmax = max(max(r["mem_req_max_b"], r["peak_rss_max_b"]) for r in rows) or 1.0
    bar_w = _PANEL_W - _LABEL_W - _VALUE_W
    parts = []
    for i, r in enumerate(rows):
        y = i * _ROW_H
        req, obs = r["mem_req_max_b"], r["peak_rss_max_b"]
        pct = 100.0 * obs / req
        over = " over" if obs > req else ""
        parts.append(
            f"<text x='0' y='{y + 15}' font-size='12' fill='#333'>"
            f"{_esc(short_process(r['process']))}</text>"
            f"<rect class='req' x='{_LABEL_W}' y='{y + 4}' "
            f"width='{max(1.0, bar_w * req / vmax):.1f}' height='{_ROW_H - 8}' "
            f"fill='#dfe4ea'>"
            f"<title>{_esc(r['process'])}: requested {fmt_bytes(req)}</title></rect>"
            f"<rect class='obs{over}' x='{_LABEL_W}' y='{y + 7}' "
            f"width='{max(1.0, bar_w * obs / vmax):.1f}' height='{_ROW_H - 14}' "
            f"fill='{'#c0392b' if over else '#2c3e50'}'>"
            f"<title>{_esc(r['process'])}: peak RSS {fmt_bytes(obs)} "
            f"({pct:.0f}% of the request)</title></rect>"
            f"<text x='{_LABEL_W + bar_w + 8}' y='{y + 15}' font-size='12' "
            f"fill='{'#c0392b' if over else '#666'}'>{pct:.0f}% of "
            f"{_esc(fmt_bytes(req))}</text>"
        )
    return _svg(
        _PANEL_W,
        len(rows) * _ROW_H + 4,
        "".join(parts),
        "Requested memory versus observed peak RSS, per process",
    )


def failure_cost_svg(roll, top=15):
    """What the run's failures and retries actually cost, per process.

    Ranked by reserved memory-hours, not by failure count: a retried task holds its
    whole reservation for its whole wall-time and delivers nothing. A process with a
    non-finite `failed_gb_h` or `failed_realtime_s` is skipped.
    """
    rows = [
        r
        for r in roll
        if r.get("n_failed")
        and (
            r.get("failed_gb_h") is None
            or _finite_or_none(r.get("failed_gb_h")) is not None
        )
        and (
            r.get("failed_realtime_s") is None
            or _finite_or_none(r.get("failed_realtime_s")) is not None
        )
    ]
    rows.sort(key=lambda r: -(r.get("failed_gb_h") or 0.0))
    rows = rows[:top]
    if not rows:
        return ""
    vmax = max((r.get("failed_gb_h") or 0.0) for r in rows) or 1.0
    bar_w = _PANEL_W - _LABEL_W - _VALUE_W
    parts = []
    for i, r in enumerate(rows):
        y = i * _ROW_H
        cost = r.get("failed_gb_h") or 0.0
        parts.append(
            f"<text x='0' y='{y + 15}' font-size='12' fill='#333'>"
            f"{_esc(short_process(r['process']))}</text>"
            f"<rect class='fail' x='{_LABEL_W}' y='{y + 4}' "
            f"width='{max(1.0, bar_w * cost / vmax):.1f}' height='{_ROW_H - 8}' "
            f"fill='#c0392b'>"
            f"<title>{_esc(r['process'])}: {r['n_failed']} failed task(s), "
            f"{fmt_secs(r.get('failed_realtime_s'))} of wall-time holding "
            f"{cost:.0f} reserved GB-hours</title></rect>"
            f"<text x='{_LABEL_W + bar_w + 8}' y='{y + 15}' font-size='12' "
            f"fill='#c0392b'>{r['n_failed']} failed, {cost:.0f} GB&#183;h</text>"
        )
    return _svg(
        _PANEL_W,
        len(rows) * _ROW_H + 4,
        "".join(parts),
        "Reserved memory-hours lost to failed and retried tasks",
    )


# Deterministic, colour-blind-safe-ish series colours, assigned by first
# appearance so the same run always paints the same process the same colour.
_SERIES = (
    "#2c3e50",
    "#c0392b",
    "#16a085",
    "#8e44ad",
    "#d35400",
    "#2980b9",
    "#7f8c8d",
    "#27ae60",
)


def _stride_sample(items, cap):
    """Deterministically thin `items` to at most `cap` entries.

    A fixed stride keeps re-renders byte-identical and spreads the sample across the
    whole series.
    """
    n = len(items)
    if n <= cap:
        return items
    stride = math.ceil(n / cap)
    return items[::stride]


def size_vs_runtime_svg(joined, top_processes=8, max_points_per_series=2000):
    """Input size against runtime, log-log, one series per process.

    Reads join_size's output (input-size log joined to the trace by process and
    tag). Log-log because inputs span MB to hundreds of GB and runtimes seconds to
    hours. Points that cannot be placed on a log axis (zero, negative or non-finite,
    see `_positive_finite`) are dropped and counted in the caption.

    The legend and point budget are ranked by point count, not first appearance: the
    top `top_processes` get a legend entry and colour, the rest share one grey
    "other (K processes)" row. Each series is capped to `max_points_per_series` by
    `_stride_sample`, and the caption says when the cap reduced the count.
    """
    usable = [
        j
        for j in joined
        if _positive_finite(j.get("input_bytes"))
        and _positive_finite(j.get("realtime_s"))
    ]
    dropped = sum(
        1
        for j in joined
        if j.get("input_bytes") is not None
        and not (
            _positive_finite(j.get("input_bytes"))
            and _positive_finite(j.get("realtime_s"))
        )
    )
    if not usable:
        if dropped:
            return (
                "<p class='empty'>"
                f"{dropped} task{'s' if dropped != 1 else ''} matched a size "
                "log but could not be plotted (zero input size or runtime "
                "cannot be placed on a log axis).</p>"
            )
        return ""

    pad_l, pad_b, pad_t, pad_r = 70, 40, 12, 200
    w, h = _PANEL_W, 380
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b

    xs_all = [math.log10(j["input_bytes"]) for j in usable]
    ys_all = [math.log10(j["realtime_s"]) for j in usable]
    x0, x1 = min(xs_all), max(xs_all)
    y0, y1 = min(ys_all), max(ys_all)
    xr = (x1 - x0) or 1.0
    yr = (y1 - y0) or 1.0

    # Group by process (keeping arrival order within each group, so the stride
    # subsample is deterministic), then rank groups by point count.
    by_proc = {}
    for j, lx, ly in zip(usable, xs_all, ys_all):
        by_proc.setdefault(j["process"], []).append((j, lx, ly))
    order = sorted(by_proc, key=lambda p: -len(by_proc[p]))
    top = order[:top_processes]
    rest = order[top_processes:]
    colour = {p: _SERIES[i % len(_SERIES)] for i, p in enumerate(top)}
    _OTHER_COLOUR = "#bdc3c7"

    parts = [
        f"<rect x='{pad_l}' y='{pad_t}' width='{plot_w}' height='{plot_h}' "
        f"fill='#fbfcfd' stroke='#e3e7ea'/>"
    ]
    total_shown = 0
    for proc in order:
        pts = _stride_sample(by_proc[proc], max_points_per_series)
        total_shown += len(pts)
        fill = colour.get(proc, _OTHER_COLOUR)
        for j, lx, ly in pts:
            cx = pad_l + plot_w * (lx - x0) / xr
            cy = pad_t + plot_h * (1.0 - (ly - y0) / yr)
            parts.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4' fill-opacity='0.7' "
                f"fill='{fill}'>"
                # short_process, not the qualified name: repeated in every <title>, the
                # prefix dominated the panel's size. The legend maps short names to colours.
                f"<title>{_esc(short_process(j['process']))} "
                f"[{_esc(j.get('tag', ''))}]: "
                f"{fmt_bytes(j['input_bytes'])} in {fmt_secs(j['realtime_s'])}"
                f"</title></circle>"
            )

    parts.append(
        f"<text x='{pad_l}' y='{h - 10}' font-size='11' fill='#666'>"
        f"input {_esc(fmt_bytes(10**x0))}</text>"
        f"<text x='{pad_l + plot_w}' y='{h - 10}' font-size='11' fill='#666' "
        f"text-anchor='end'>{_esc(fmt_bytes(10**x1))}</text>"
        f"<text x='{pad_l - 6}' y='{pad_t + 10}' font-size='11' fill='#666' "
        f"text-anchor='end'>{_esc(fmt_secs(10**y1))}</text>"
        f"<text x='{pad_l - 6}' y='{pad_t + plot_h}' font-size='11' fill='#666' "
        f"text-anchor='end'>{_esc(fmt_secs(10**y0))}</text>"
    )

    for i, proc in enumerate(top):
        ly = pad_t + i * 18
        parts.append(
            f"<rect class='key' x='{pad_l + plot_w + 16}' y='{ly}' width='10' "
            f"height='10' fill='{colour[proc]}'/>"
            f"<text x='{pad_l + plot_w + 32}' y='{ly + 9}' font-size='11' "
            f"fill='#333'>{_esc(short_process(proc))}</text>"
        )
    if rest:
        ly = pad_t + len(top) * 18
        parts.append(
            f"<rect class='key' x='{pad_l + plot_w + 16}' y='{ly}' width='10' "
            f"height='10' fill='{_OTHER_COLOUR}'/>"
            f"<text x='{pad_l + plot_w + 32}' y='{ly + 9}' font-size='11' "
            f"fill='#333'>other ({len(rest)} process"
            f"{'es' if len(rest) != 1 else ''})</text>"
        )

    if dropped:
        parts.append(
            f"<text x='{pad_l}' y='{pad_t - 2}' font-size='11' fill='#888'>"
            f"{dropped} task not shown (zero input size or runtime cannot be "
            f"placed on a log axis)</text>"
            if dropped == 1
            else f"<text x='{pad_l}' y='{pad_t - 2}' font-size='11' fill='#888'>"
            f"{dropped} tasks not shown (zero input size or runtime cannot be "
            f"placed on a log axis)</text>"
        )
    if total_shown < len(usable):
        parts.append(
            f"<text x='{pad_l}' y='{h - 24}' font-size='11' fill='#888'>"
            f"{total_shown} of {len(usable)} points shown (each process "
            f"capped to {max_points_per_series})</text>"
        )

    title = (
        "Input size against runtime, log-log, one series per process"
        if not rest
        else (
            "Input size against runtime, log-log; legend shows the "
            f"{len(top)} process(es) with the most points, {len(rest)} more "
            'grouped as "other"'
        )
    )
    return _svg(w, h, "".join(parts), title)


_CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;background:#f4f6f9;color:#333;margin:0}
header{background:#1a2332;color:#fff;padding:24px 40px}
main{max-width:1400px;margin:32px auto;padding:0 24px 60px}
section{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1);margin-bottom:32px;overflow:hidden}
h2{background:#2c3e50;color:#fff;padding:14px 20px;font-size:1.15rem;margin:0}
.body{padding:20px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th{background:#ecf0f1;text-align:left;padding:8px 12px;border-bottom:2px solid #bdc3c7}
td{padding:7px 12px;border-bottom:1px solid #ecf0f1}
tr:hover td{background:#f8f9fa}
.empty{color:#888;font-style:italic}
.fail{color:#c0392b;font-weight:600}
svg{margin:4px 0}
svg text{font-family:'Segoe UI',Arial,sans-serif}
"""


def _esc(v):
    """Escape a dynamic value (process name, tag, status, exit code, path,
    etc.) for safe inclusion in HTML. Kept local so this script stays
    stdlib-only and self-contained."""
    return html.escape(str(v))


def _section(title, body):
    return f"<section><h2>{title}</h2><div class='body'>{body}</div></section>"


def build_html(
    trace_rows, size_map, timestamp, native_report=None, native_timeline=None
):
    """Render the whole self-contained resource report as one HTML string.

    Parameters
    ----------
    trace_rows : list of dict
        Rows parsed from Nextflow's ``trace.txt``, already joined with input sizes.
    size_map : dict
        Process/sample key -> input bytes, from the aggregated size log.
    timestamp : str
        Generation time, rendered into the header.
    native_report, native_timeline : str, optional
        Paths to Nextflow's own ``report.html`` / ``timeline.html``, linked when given.

    Returns
    -------
    str
        A complete HTML document with the CSS inlined; no external assets.
    """
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>MIRAGE Resource Report</title><style>{_CSS}</style></head><body>",
        f"<header><h1>MIRAGE Computational Resource Report</h1>"
        f"<div style='opacity:.75;font-size:.9rem'>Generated: {timestamp}</div></header><main>",
    ]

    if not trace_rows:
        parts.append(
            _section(
                "Resource Usage",
                "<p class='empty'>Trace data not available "
                "(enable_trace is a boolean param -- turn it on with a "
                "profile or -params-file to collect it).</p>",
            )
        )
        parts.append("</main></body></html>")
        return "".join(parts)

    # Run totals table; everything below it is a panel.
    total_wall = _sumf([r.get("realtime_s") for r in trace_rows])
    n_fail = sum(1 for r in trace_rows if _is_failure(r))
    peak = _maxf([r.get("peak_rss_b") for r in trace_rows])
    peak_cpu = _maxf([r.get("cpu_pct") for r in trace_rows])
    # `None` (no reading) is "N/A"; a non-finite reading is "n/a", as in
    # fmt_bytes/fmt_secs.
    if peak_cpu is None:
        peak_cpu_cell = "N/A"
    elif _finite_or_none(peak_cpu) is None:
        peak_cpu_cell = "n/a"
    else:
        peak_cpu_cell = f"{peak_cpu}%"
    totals = (
        f"<table><tbody>"
        f"<tr><th>Total tasks</th><td>{len(trace_rows)}</td></tr>"
        f"<tr><th>Total CPU wall-time</th><td>{fmt_secs(total_wall)}</td></tr>"
        f"<tr><th>Failed/non-zero exit</th><td>{n_fail}</td></tr>"
        f"<tr><th>Peak single-task RSS</th><td>{fmt_bytes(peak)}</td></tr>"
        f"<tr><th>Peak single-task %CPU</th><td>{peak_cpu_cell}</td></tr>"
        f"</tbody></table>"
    )
    parts.append(_section("Run Totals", totals))

    roll = rollup_by_process(trace_rows)

    # 1. Where the wall-time went.
    parts.append(
        _section(
            "Wall-time by Process",
            walltime_bars_svg(roll)
            or "<p class='empty'>No task recorded a runtime.</p>",
        )
    )

    # 2. Requested vs observed memory -- where over-provisioning lives.
    parts.append(
        _section(
            "Memory Headroom &mdash; Requested vs Observed",
            memory_headroom_svg(roll)
            or "<p class='empty'>No task recorded both a memory request and a "
            "peak RSS, so there is no headroom to show.</p>",
        )
    )

    # 3. What failures and retries cost, in reserved memory-hours.
    parts.append(
        _section(
            "Retries &amp; Failures",
            failure_cost_svg(roll)
            or "<p class='empty'>No failed or non-zero-exit tasks.</p>",
        )
    )

    # 4. Does cost scale with the data?
    parts.append(
        _section(
            "Input Size vs Runtime",
            size_vs_runtime_svg(join_size(trace_rows, size_map))
            or "<p class='empty'>No size logs matched trace tasks.</p>",
        )
    )

    # Pointers to native reports
    links = []
    if native_report:
        links.append(
            f"<li>Interactive execution report: <code>{_esc(native_report)}</code></li>"
        )
    if native_timeline:
        links.append(f"<li>Timeline: <code>{_esc(native_timeline)}</code></li>")
    if links:
        parts.append(
            _section("Nextflow Native Reports", "<ul>" + "".join(links) + "</ul>")
        )

    parts.append("</main></body></html>")
    return "".join(parts)


def parse_args():
    """Parse the resource-report CLI.

    Returns
    -------
    argparse.Namespace
        ``trace`` (default ``.trace/trace.txt``), ``size_log``, ``output``,
        ``native_report``, ``native_timeline``.
    """
    p = argparse.ArgumentParser(
        description="Generate MIRAGE computational-resource report"
    )
    p.add_argument("--trace", default=".trace/trace.txt")
    p.add_argument("--size-log", default=None)
    p.add_argument("--output", default="mirage_resource_report.html")
    p.add_argument("--native-report", default=None)
    p.add_argument("--native-timeline", default=None)
    return p.parse_args()


def main():
    """CLI entry point: join the trace with the size log and write the HTML report.

    Returns
    -------
    int
        0 on success, including when the trace path does not exist. 3 when the
        trace exists but has zero task rows (header-only), so main.nf logs a
        warning instead of announcing an empty report.
    """
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    trace_path_exists = Path(args.trace).exists() if args.trace else False
    trace_rows = parse_trace(args.trace)
    size_map = parse_size_log(args.size_log)
    # Named report_html (not `html`) to avoid shadowing the stdlib `html`
    # module used for escaping elsewhere in this file.
    report_html = build_html(
        trace_rows, size_map, timestamp, args.native_report, args.native_timeline
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_html, encoding="utf-8")
    print(f"Resource report written to: {out}")
    if trace_path_exists and not trace_rows:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
