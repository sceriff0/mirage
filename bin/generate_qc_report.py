#!/usr/bin/env python3
"""
generate_qc_report.py
Generates a self-contained HTML QC report for a MIRAGE pipeline run.
All images are embedded as base64; no external dependencies required.
"""

import argparse
import base64
import csv
import html
import json
import logging
import math
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from logger import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Generate MIRAGE QC HTML report")
    p.add_argument(
        "--preprocess-qc",
        default="preprocess_qc/",
        help="Directory of preprocessing QC PNGs",
    )
    p.add_argument(
        "--registration-qc",
        default="registration_qc/",
        help="Directory of registration QC PNGs",
    )
    p.add_argument(
        "--registration-tre",
        default="registration_tre/",
        help=(
            "Directory of the registration method's own target-registration-error "
            "artifacts (VALIS *_summary.csv / STARE *_tre.json)"
        ),
    )
    p.add_argument(
        "--postprocess-qc",
        default="postprocess_qc/",
        help="Directory of postprocessing QC PNGs",
    )
    p.add_argument(
        "--versions",
        default=None,
        help=(
            "Path to the collated versions.yml. NOT rendered in the report: it is "
            "copied verbatim into --data-dir as collated_versions.yml, which is "
            "where a machine-readable version record belongs. A per-process "
            "software table is 200 rows nobody reads and a parser nobody trusts."
        ),
    )
    p.add_argument("--run-summary", default=None, help="Path to run_summary.json")
    p.add_argument(
        "--seg-qc",
        default="seg_qc/",
        help="Directory of warp-segmentation QC JSONs",
    )
    p.add_argument(
        "--seg-residuals",
        default="seg_residuals/",
        help="Directory of per-cell registration residual CSVs (SEG_QC.out.per_cell)",
    )
    p.add_argument("--output", default="mirage_qc_report.html", help="Output HTML path")
    p.add_argument(
        "--data-dir",
        default="mirage_qc_data/",
        help="Directory to copy input data into",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_files(directory, pattern="*"):
    """Return sorted list of existing files matching glob pattern inside directory."""
    if not directory:
        return []
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob(pattern))


def _img_to_b64(path):
    """Return a data URI string for an image file (PNG or TIFF)."""
    ext = Path(path).suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    mime = mime_types.get(ext, "image/png")
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_csv_table(csv_path):
    """
    Parse a generic CSV into (headers, rows) where each row is a list of
    strings in header order. Used for the Valis preprocessed_summary.csv
    (rTRE, D, n_matches, etc.); kept generic (columns are whatever the CSV
    declares) so any future summary CSV can reuse it unchanged.
    """
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        display_headers = reader.fieldnames or []
        rows = []
        for row in reader:
            rows.append([row.get(h, "") for h in display_headers])
    return display_headers, rows


def parse_run_summary_json(path):
    """Load the workflow-written run_summary.json; return {} on any problem."""
    if not path or not Path(path).exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _kv_table(pairs):
    """Render a list of (label, value) tuples as a two-column table."""
    rows = "".join(
        f"<tr><th style='width:240px'>{html.escape(str(k))}</th>"
        f"<td>{'' if v is None else html.escape(str(v))}</td></tr>"
        for k, v in pairs
    )
    return f"<table><tbody>{rows}</tbody></table>"


def run_summary_section(summary):
    """Top overview card: pipeline, run context, and key parameters used."""
    if not summary:
        return section(
            "Run Summary", '<p class="empty-notice">Run summary not available.</p>'
        )
    pipe = summary.get("pipeline", {})
    run = summary.get("run", {})
    params = summary.get("params", {})
    pairs = [
        ("Pipeline", f"{pipe.get('name', 'mirage')} v{pipe.get('version', '?')}"),
        ("Run timestamp", run.get("timestamp")),
        ("Mode", run.get("mode")),
        ("Steps", f"{run.get('start', '?')} → {run.get('stop', '?')}"),
    ]
    for key in sorted(params):
        pairs.append((f"param: {key}", params[key]))
    return section("Run Summary", _kv_table(pairs))


def status_strip_section(present):
    """A row of stage badges: ran (green) vs no artifacts (grey)."""
    badges = []
    for stage, ran in present.items():
        color = "#27ae60" if ran else "#95a5a6"
        label = "ran" if ran else "no artifacts"
        badges.append(
            f"<span style='display:inline-block;margin:4px 8px 4px 0;padding:6px 12px;"
            f"border-radius:14px;background:{color};color:#fff;font-size:0.85rem;'>"
            f"{html.escape(str(stage))}: {label}</span>"
        )
    return section("Pipeline Stages", "<div>" + "".join(badges) + "</div>")


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f4f6f9;
    color: #333;
    line-height: 1.5;
}
header {
    background: #1a2332;
    color: #fff;
    padding: 24px 40px;
}
header h1 { font-size: 2rem; font-weight: 700; }
header .subtitle { font-size: 0.95rem; opacity: 0.75; margin-top: 4px; }
main { max-width: 1400px; margin: 32px auto; padding: 0 24px 60px; }
section { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
          margin-bottom: 32px; overflow: hidden; }
section h2 {
    background: #2c3e50;
    color: #fff;
    padding: 14px 20px;
    font-size: 1.15rem;
    font-weight: 600;
}
.section-body { padding: 20px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th { background: #ecf0f1; text-align: left; padding: 8px 12px; font-weight: 600;
     border-bottom: 2px solid #bdc3c7; }
td { padding: 7px 12px; border-bottom: 1px solid #ecf0f1; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8f9fa; }
.img-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 8px;
}
.img-grid-wide {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 20px;
    margin-top: 8px;
}
.img-card {
    border: 1px solid #dde2ea;
    border-radius: 6px;
    overflow: hidden;
    background: #fafbfc;
}
.img-card img { width: 100%; display: block; }
.img-card .caption {
    font-size: 0.78rem;
    padding: 6px 8px;
    text-align: center;
    color: #555;
    word-break: break-all;
}
.empty-notice {
    color: #888;
    font-style: italic;
    padding: 8px 0;
}
"""


def html_header(timestamp):
    """Return the HTML document header, including the title and generation timestamp."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MIRAGE QC Report</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>MIRAGE QC Report</h1>
  <div class="subtitle">Generated: {timestamp}</div>
</header>
<main>
"""


def html_footer():
    """Return the closing HTML tags for the report."""
    return "</main>\n</body>\n</html>\n"


def section(title, body_html):
    """Wrap ``body_html`` in a titled report section."""
    return (
        f"<section>\n"
        f"  <h2>{title}</h2>\n"
        f'  <div class="section-body">\n{body_html}\n  </div>\n'
        f"</section>\n"
    )


def _html_table(headers, rows, extra_body_html=""):
    """
    Build a complete ``<table>`` (with ``<thead>``/``<tbody>``), escaping
    every header and every cell value via ``html.escape``.

    Centralizes the table-render logic that used to be duplicated across the
    registration-QC Valis rTRE table and the STARE tiled-TRE table, and
    ensures neither can inject raw CSV/JSON-derived HTML into the report.

    `extra_body_html`: optional pre-rendered ``<tr>...</tr>`` HTML appended
    after the normal rows — for full-width/``colspan`` rows (e.g. a
    per-file parse-error notice) that don't fit the uniform one-cell-per-
    header row shape every other row uses. The caller is responsible for
    escaping any dynamic values it interpolates into this HTML itself.
    """
    thead = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    body += extra_body_html
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>"


# ── inline SVG plotting ─────────────────────────────────────────────────────────
# Hand-rolled and stdlib-only, following this file's existing
# `_tiled_tre_heatmap_svg()` precedent. bin/generate_resource_report.py hand-rolls
# its own SVG for the same reason, and the two deliberately do NOT share a module:
# that script runs on the head node OUTSIDE every container, this one runs inside
# bolt3x/mirage-preprocess, so a shared bin/utils module would have to be staged
# into both and would couple two independently released surfaces. Sixty duplicated
# lines are cheaper than that.

SVG_W = 380  # plot box width, px
SVG_H = 170  # plot box height, px
SVG_PAD_L = 46  # left gutter: y-axis (count) labels
SVG_PAD_R = 12
SVG_PAD_T = 22  # top gutter: the plot's own title
SVG_PAD_B = 32  # bottom gutter: x-axis labels


def _finite(values):
    """Keep only real, finite numbers from ``values``.

    ``bool`` is excluded on purpose: it is an ``int`` subclass, so a stray
    ``True`` in a metrics dict would otherwise be plotted as 1.0. NaN AND
    ``±inf`` are both dropped -- ``json.loads`` accepts a literal
    ``Infinity``, and a zero-denominator QC metric can produce one, so this
    is a realistic input, not a hypothetical one. An unfiltered inf would
    become ``lo``/``hi`` in ``_bin_counts``, making its span (and then
    ``int(span)``) NaN.
    """
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if not math.isfinite(v):  # NaN, +inf, -inf
            continue
        out.append(float(v))
    return out


def _percentile(values, q):
    """Nearest-rank percentile (``q`` in 0..100) of the finite part of ``values``.

    Nearest-rank rather than interpolated: every number this plots is an error
    measured in pixels or µm, and an interpolated p90 invents a value no tile and
    no cell ever had. ``None`` when nothing is finite.
    """
    vals = sorted(_finite(values))
    if not vals:
        return None
    k = max(1, min(len(vals), int(math.ceil(q / 100.0 * len(vals)))))
    return vals[k - 1]


def _bin_counts(values, bins):
    """Bin the finite part of ``values`` into ``bins`` equal-width buckets.

    Returns ``(counts, lo, hi)``, or ``([], 0.0, 0.0)`` when there is nothing to
    bin. Three edge cases are handled here rather than in every caller: the maximum
    value lands in the LAST bin (``int((hi - lo) / span * bins)`` is ``bins``,
    one past the end); an all-identical input widens to ``lo + 1.0`` so the
    single bar has a width to be drawn in instead of dividing by zero; and a span
    that overflows to ``inf`` (both ``lo``/``hi`` finite but far enough apart)
    makes the normalized position ``nan`` -- that also lands in the last bin
    rather than raising out of ``int()``.
    """
    vals = _finite(values)
    if not vals or bins < 1:
        return [], 0.0, 0.0
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1.0
    counts = [0] * bins
    span = hi - lo
    for v in vals:
        pos = (v - lo) / span
        idx = bins - 1 if not math.isfinite(pos) else min(int(pos * bins), bins - 1)
        counts[idx] += 1
    return counts, lo, hi


def _histogram_svg(values, *, title, x_label, bins=20, rules=()):
    """One binned distribution as an inline ``<svg>`` bar chart.

    ``rules`` is an iterable of ``(value, label)`` drawn as dashed vertical
    markers carrying ``class="rule"`` — the p50/p90 lines the error-distribution
    view adds. A rule whose value is missing or outside the data range is
    skipped, never crashed on: these plots render whatever a partially-failed run
    produced.

    Returns the shared ``empty-notice`` paragraph when nothing is finite, so
    every caller gets the same "no data" affordance without repeating the check.
    """
    counts, lo, hi = _bin_counts(values, bins)
    if not counts:
        return (
            f'<p class="empty-notice">{html.escape(str(title))}: no values to plot.</p>'
        )
    n = sum(counts)
    cmax = max(counts) or 1
    pw = SVG_W - SVG_PAD_L - SVG_PAD_R
    ph = SVG_H - SVG_PAD_T - SVG_PAD_B
    bw = pw / float(bins)
    base = SVG_PAD_T + ph
    parts = [
        f'<svg width="{SVG_W}" height="{SVG_H}" role="img" '
        f'aria-label="{html.escape(str(title))}" style="border:1px solid #eee;">',
        f'<text x="{SVG_PAD_L}" y="14" font-size="11" fill="#444">'
        f"{html.escape(str(title))} (n={n})</text>",
    ]
    for i, c in enumerate(counts):
        if not c:
            continue
        bh = ph * c / cmax
        edge_lo = lo + (hi - lo) * i / bins
        edge_hi = lo + (hi - lo) * (i + 1) / bins
        parts.append(
            f'<rect x="{SVG_PAD_L + i * bw:.2f}" y="{base - bh:.2f}" '
            f'width="{max(bw - 1.0, 0.5):.2f}" height="{bh:.2f}" fill="#4a7fb5">'
            f"<title>{edge_lo:.3g}–{edge_hi:.3g}: {c}</title></rect>"
        )
    for value, label in rules:
        v = _finite([value])
        if not v or not (lo <= v[0] <= hi):
            continue
        x = SVG_PAD_L + pw * (v[0] - lo) / (hi - lo)
        parts.append(
            f'<line class="rule" x1="{x:.2f}" y1="{SVG_PAD_T}" x2="{x:.2f}" '
            f'y2="{base}" stroke="#c0392b" stroke-width="1" stroke-dasharray="3,2">'
            f"<title>{html.escape(str(label))}: {v[0]:.3g}</title></line>"
        )
    parts.append(
        f'<line x1="{SVG_PAD_L}" y1="{base}" x2="{SVG_PAD_L + pw}" y2="{base}" '
        'stroke="#999"/>'
    )
    parts.append(
        f'<line x1="{SVG_PAD_L}" y1="{SVG_PAD_T}" x2="{SVG_PAD_L}" y2="{base}" '
        'stroke="#999"/>'
    )
    parts.append(
        f'<text x="{SVG_PAD_L}" y="{SVG_H - 18}" font-size="9" fill="#666">'
        f"{lo:.3g}</text>"
    )
    parts.append(
        f'<text x="{SVG_PAD_L + pw}" y="{SVG_H - 18}" font-size="9" fill="#666" '
        f'text-anchor="end">{hi:.3g}</text>'
    )
    parts.append(
        f'<text x="{SVG_PAD_L + pw / 2:.0f}" y="{SVG_H - 5}" font-size="10" '
        f'fill="#444" text-anchor="middle">{html.escape(str(x_label))}</text>'
    )
    parts.append(
        f'<text x="10" y="{SVG_PAD_T + 8}" font-size="9" fill="#666">{cmax}</text>'
    )
    parts.append(f'<text x="10" y="{base}" font-size="9" fill="#666">0</text>')
    parts.append("</svg>")
    return "".join(parts)


def _error_distribution_svg(values, *, title, bins=20):
    """A stage's error distribution: the histogram plus its p50 and p90 markers.

    A registration-error distribution is read by its TAIL — the median says the
    run was fine, the p90 says where it was not — so both percentiles are drawn
    on the plot rather than left to be eyeballed off the bars. Units belong in
    ``title`` (px for per-tile TRE, µm for cell displacement); the x-axis is
    labelled ``error`` because one function serves both.
    """
    rules = [
        (v, lbl)
        for v, lbl in (
            (_percentile(values, 50), "p50"),
            (_percentile(values, 90), "p90"),
        )
        if v is not None
    ]
    return _histogram_svg(values, title=title, x_label="error", bins=bins, rules=rules)


def img_grid(png_files, wide=False):
    """Render a list of PNG files as a grid of base64-embedded image cards."""
    if not png_files:
        return '<p class="empty-notice">No images found.</p>'
    grid_class = "img-grid-wide" if wide else "img-grid"
    cards = []
    for p in png_files:
        b64 = _img_to_b64(p)
        name = html.escape(Path(p).name)
        cards.append(
            f'<div class="img-card">'
            f'<img src="{b64}" alt="{name}" loading="lazy">'
            f'<div class="caption">{name}</div>'
            f"</div>"
        )
    return f'<div class="{grid_class}">' + "".join(cards) + "</div>"


def preprocess_qc_section(preprocess_dir):
    """Build the preprocessing-QC report section from the PNGs in ``preprocess_dir``."""
    pngs = list_files(preprocess_dir, "*.png")
    return section("Preprocessing QC", img_grid(pngs))


def registration_qc_section(reg_dir, valis_dir, seg_qc_dir=None):
    """Build the registration-QC report section (overlay images plus accuracy tables)."""
    parts = []

    # Image grid
    pngs = list_files(reg_dir, "*.png")
    parts.append(
        "<h3 style='margin-bottom:8px;font-size:1rem;color:#444;'>Overlay Images</h3>"
    )
    parts.append(img_grid(pngs))

    # Registration-accuracy summaries. The method decides the format: VALIS emits rTRE CSVs, the
    # tiled ('STARE') method emits its own intrinsic-TRE JSON (_tre.json) — both land here.
    valis_csvs = list_files(valis_dir, "*.csv")
    tiled_tre_jsons = list_files(valis_dir, "*_tre.json")
    if valis_csvs:
        parts.append(
            "<h3 style='margin:20px 0 8px;font-size:1rem;color:#444;'>Registration Accuracy (Valis rTRE)</h3>"
        )
        for csv_path in valis_csvs:
            try:
                headers, rows = parse_csv_table(csv_path)
            except Exception as exc:
                parts.append(
                    '<p class="empty-notice">Could not parse '
                    f"{html.escape(Path(csv_path).name)}: {html.escape(str(exc))}</p>"
                )
                continue
            parts.append(
                "<p style='font-size:0.85rem;color:#666;margin-bottom:6px;'>"
                f"{html.escape(Path(csv_path).name)}</p>"
            )
            parts.append(_html_table(headers, rows))
    if tiled_tre_jsons:
        parts.append(
            "<h3 style='margin:20px 0 8px;font-size:1rem;color:#444;'>Registration Accuracy "
            "(STARE Tiled TRE)</h3>"
        )
        parts.append(_tiled_tre_plots(tiled_tre_jsons))
    if not valis_csvs and not tiled_tre_jsons:
        parts.append(
            '<p class="empty-notice" style="margin-top:12px;">No registration-accuracy summary found.</p>'
        )

    # Warp-segmentation QC metrics (previously dropped)
    seg_qc_jsons = list_files(seg_qc_dir, "*.json") if seg_qc_dir else []
    if seg_qc_jsons:
        parts.append(
            "<h3 style='margin:20px 0 8px;font-size:1rem;color:#444;'>Warp-Segmentation QC</h3>"
        )
        parts.append(_seg_qc_stage_plots(seg_qc_dir))

    return section("Registration QC", "\n".join(parts))


def postprocess_qc_section(postprocess_dir):
    """Build the postprocessing-QC report section (histograms/stats, excluding seg overlays)."""
    # Exclude seg_overlay images, keep only histograms/stats
    all_pngs = list_files(postprocess_dir, "*.png")
    pngs = [p for p in all_pngs if "_seg_overlay" not in p.name]
    return section("Postprocessing QC", img_grid(pngs, wide=True))


# Canonical stage order for the warp-seg QC plots. WARP_SEG_QC writes `stage_order`
# into its JSON, but a directory holding several slides can hold several orders and
# a report is diffed across runs — so the order is fixed here and anything
# unrecognised is appended, sorted, rather than dropped. `refined` is the tiled
# (STARE) backend's terminal stage (bin/utils/tiled_stage_warp.py's STAGES =
# native, rigid, refined -- it never emits non_rigid/micro), placed last as the
# tiled analogue of VALIS's `micro`: the final measured stage.
SEG_QC_STAGE_ORDER = ("native", "rigid", "non_rigid", "micro", "refined")


def _seg_qc_stage_plots(seg_qc_dir):
    """Per-stage cell-displacement distribution across every slide's WARP_SEG_QC JSON.

    This replaced a per-file flattened metric table (spec Phase 4). The JSON carries
    per-stage SUMMARY statistics, not raw per-cell values, so the distribution here
    is across SLIDES: one point per slide per stage, the ``displacement_um_p50``
    (px fallback) read through ``_read_seg_cell_disp`` — the same reader
    ``reconcile_rows`` uses, so the plot and the reconciliation scatter cannot
    disagree about a number. On a one-slide run this is a single bar whose ``n=1``
    says so; the per-slide records are copied verbatim into the report's
    ``seg_qc/`` data folder, which is where a single slide's numbers belong.

    Units are never mixed within one distribution. STARE (tiled) runs are the
    COMMON case for px, not an edge case (see ``_read_seg_cell_disp``), so a
    directory holding both a calibrated (µm) and an uncalibrated (px) slide for
    the same stage is split into up to two plots, one per unit actually present,
    each titled with its own unit rather than a hardcoded one.

    The caption also reports the matched-cell count (``matching.n_pairs``, via
    ``_read_seg_qc_match_counts``) across slides, min and median: a p50 measured
    over 12 matched cells is a different claim from one measured over 40 000, and
    the histogram alone cannot tell the two apart.
    """
    per_slide = _read_seg_cell_disp(seg_qc_dir)
    if not per_slide:
        return '<p class="empty-notice">No warp-segmentation QC metrics found.</p>'
    seen = set()
    for metrics in per_slide.values():
        seen |= set(metrics)
    stages = [s for s in SEG_QC_STAGE_ORDER if s in seen]
    stages += sorted(seen - set(SEG_QC_STAGE_ORDER))
    unit_labels = {"um": "µm", "px": "px"}
    parts = ['<div style="display:flex;flex-wrap:wrap;gap:12px;">']
    for stage in stages:
        by_unit = {}
        for metrics in per_slide.values():
            entry = metrics.get(stage)
            if entry is None:
                continue
            val, unit = entry
            if val is not None:
                by_unit.setdefault(unit, []).append(val)
        # Fixed unit order (um before px) so a mixed directory renders deterministically.
        for unit in ("um", "px"):
            vals = by_unit.get(unit)
            if not vals:
                continue
            parts.append(
                _error_distribution_svg(
                    vals,
                    title=(
                        f"{stage} — cell displacement p50 ({unit_labels[unit]}), "
                        "one point per slide"
                    ),
                )
            )
    parts.append("</div>")
    caption = f"{len(per_slide)} slide(s)."
    match_counts = _read_seg_qc_match_counts(seg_qc_dir)
    if match_counts:
        counts = sorted(match_counts.values())
        caption += (
            f" Matched cells per slide: min {counts[0]}, "
            f"median {statistics.median(counts):g}."
        )
    caption += (
        " Per-slide records are published unchanged under <code>seg_qc/</code> "
        "in this report's data folder."
    )
    parts.append(
        f"<p style='font-size:0.8rem;color:#888;margin-top:6px;'>{caption}</p>"
    )
    return "\n".join(parts)


# ── STARE tiled-registration intrinsic TRE ──────────────────────────────────────
# The tiled ('STARE') method emits its own intrinsic Target Registration Error per slide
# (_tre.json), the way VALIS emits rTRE: coarse (rigid) feature-fit TRE, a per-tile spatial
# heatmap VALIS does not have, and — in the default path — the post-refinement residual (its
# analogue of VALIS's non-rigid error). Rendered here alongside the other registration-accuracy
# tables so both methods surface a comparable number.
def _fmt_px(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"


def parse_tiled_tre_json(path):
    """Flatten a STARE _tre.json into the headline registration-accuracy fields."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    def pct(block, key):
        b = data.get(block)
        return b.get(key) if isinstance(b, dict) else None

    tiles = data.get("tiles", []) or []
    # Prefer the counts the solver wrote; fall back to counting the heatmap so a _tre.json
    # from before those keys existed still reports sensibly (missing `accepted` = accepted).
    n_rejected = data.get("n_rejected")
    if n_rejected is None:
        n_rejected = sum(1 for t in tiles if not t.get("accepted", True))
    n_accepted = data.get("n_accepted")
    if n_accepted is None:
        n_accepted = len(tiles) - n_rejected

    return {
        "file": Path(path).name,
        "moving": data.get("moving") or Path(path).name,
        "coarse_tre_px": data.get("coarse_tre_px"),
        "n_tiles": data.get("n_tiles"),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "mesh_refined": bool(data.get("mesh_refined")),
        "rigid_p50": pct("rigid_tre_px", "p50"),
        "rigid_p90": pct("rigid_tre_px", "p90"),
        "final_p50": pct("residual_after_px", "p50"),
        "final_p90": pct("residual_after_px", "p90"),
        "tiles": tiles,
    }


def _tiled_tre_heatmap_svg(info, cell=14):
    """A compact per-tile TRE heatmap (green=low, red=high) — STARE's spatial accuracy view."""
    tiles = info["tiles"]
    if not tiles:
        return ""
    have_final = all("tre_after" in t for t in tiles)

    def val(t):
        return t["tre_after"] if have_final else t.get("tre_rigid", 0.0)

    def kept(t):
        return bool(t.get("accepted", True))

    # Scale over ACCEPTED tiles only. A rejected tile's value is an artefact of correlating
    # against background — one of them at 131.5px would otherwise set vmax and squash every
    # real tile into the green end, hiding the misalignment the heatmap exists to show.
    vmax = max((val(t) for t in tiles if kept(t)), default=0.0) or 1.0
    nx = max(t["ix"] for t in tiles) + 1
    ny = max(t["iy"] for t in tiles) + 1
    rects = []
    for t in tiles:
        if kept(t):
            frac = min(val(t) / vmax, 1.0)
            r, g, b = int(60 + 195 * frac), int(180 * (1 - frac)), 70
            title = f"tile ({t['ix']},{t['iy']}): {val(t):.2f}px"
        else:
            # Neutral grey, never on the green–red scale: this tile carries no measurement.
            r, g, b = 200, 200, 200
            title = f"tile ({t['ix']},{t['iy']}): rejected — not in the mesh"
        rects.append(
            f'<rect x="{t["ix"] * cell}" y="{t["iy"] * cell}" width="{cell - 1}" '
            f'height="{cell - 1}" fill="rgb({r},{g},{b})">'
            f"<title>{title}</title></rect>"
        )
    label = "post-refinement residual" if have_final else "rigid-stage TRE"
    n_rejected = sum(1 for t in tiles if not kept(t))
    dropped = f"; {n_rejected} rejected (grey)" if n_rejected else ""
    return (
        '<div style="margin:8px 0 4px;">'
        f'<p style="font-size:0.8rem;color:#666;margin:0 0 2px;">'
        f"{html.escape(info['moving'])} — per-tile {label} "
        f"(green=low, red=high; max {vmax:.2f}px{dropped})</p>"
        f'<svg width="{nx * cell}" height="{ny * cell}" '
        f'style="border:1px solid #eee;">{"".join(rects)}</svg></div>'
    )


def _tiled_tre_plots(tre_jsons):
    """Per-slide STARE intrinsic TRE: headline caption, per-stage distributions, heatmap.

    This replaced a per-slide summary TABLE (spec Phase 4). The numbers that table
    carried — coarse TRE, rigid and final percentiles, whether the mesh was
    refined, accepted/total tiles — are kept in the caption line above each
    slide's plots: dropping a table is not licence to drop the numbers, and the
    accepted/total ratio in particular is the reason a clean-looking distribution
    can still describe a slide a third of which was never measured.

    The distributions are built from ACCEPTED tiles only, the same rule
    bin/utils/tre_report.py applies to its percentiles — a rejected tile's
    ``tre_rigid`` is an artefact of correlating against background. The heatmap
    below them still shows every tile, accepted or not, because *where* points
    were dropped is exactly what it is for.
    """
    parts, errors = [], []
    for jp in sorted(tre_jsons):
        try:
            info = parse_tiled_tre_json(jp)
            heatmap = _tiled_tre_heatmap_svg(info)
        except Exception as exc:  # noqa: BLE001 - report, never crash
            errors.append(
                '<p class="empty-notice">Could not parse '
                f"{html.escape(Path(jp).name)}: {html.escape(str(exc))}</p>"
            )
            continue
        kept = [t for t in info["tiles"] if bool(t.get("accepted", True))]
        tiles_txt = (
            f"{info['n_accepted']} / {info['n_tiles']}"
            if info["n_rejected"]
            else str(info["n_tiles"])
        )
        final = (
            f"{_fmt_px(info['final_p50'])} / {_fmt_px(info['final_p90'])}"
            if info["final_p50"] is not None
            else "n/a (fan-out — see benchmark)"
        )
        moving = str(info["moving"])
        parts.append(
            "<p style='font-size:0.85rem;color:#666;margin:14px 0 4px;'>"
            f"<b>{html.escape(moving)}</b> — coarse TRE "
            f"{_fmt_px(info['coarse_tre_px'])} px; rigid p50 / p90 "
            f"{_fmt_px(info['rigid_p50'])} / {_fmt_px(info['rigid_p90'])} px; "
            f"final p50 / p90 {final} px; mesh refined: "
            f"{'yes' if info['mesh_refined'] else 'no'}; tiles {tiles_txt}</p>"
        )
        parts.append('<div style="display:flex;flex-wrap:wrap;gap:12px;">')
        parts.append(
            _error_distribution_svg(
                [t.get("tre_rigid") for t in kept],
                title=f"{moving} — rigid stage, per-tile TRE (px)",
            )
        )
        if kept and all("tre_after" in t for t in kept):
            parts.append(
                _error_distribution_svg(
                    [t.get("tre_after") for t in kept],
                    title=f"{moving} — post-refinement residual (px)",
                )
            )
        parts.append("</div>")
        if heatmap:
            parts.append(heatmap)
    return "\n".join(parts + errors)


# ── (C) feature-TRE vs cell-displacement reconciliation ─────────────────────────
# VALIS scores registration on its own SuperPoint/SuperGlue keypoints — a self-referential,
# optimistic feature-TRE — while WARP_SEG_QC scores it on independently segmented cells. Lining
# the two up per stage exposes the failure mode neither catches alone: features declaring a slide
# aligned while the cells say otherwise. A stage counts as divergent when the two disagree by more
# than this factor; below it they are treated as corroborating.
RECONCILE_DIVERGENCE_RATIO = 3.0

# Each stage's feature-TRE comes from a specific VALIS summary column, because register_micro()
# composes the micro residual into the same field and rewrites non_rigid_D in place: only the
# pre-micro summary still isolates the non-rigid stage. (col, which_summary).
_RECONCILE_TRE_SOURCE = {
    "rigid": ("rigid_D", "final"),
    "non_rigid": ("non_rigid_D", "premicro"),
    "micro": ("non_rigid_D", "final"),
}


def _to_float(v):
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _read_intrinsic_tre(tre_dir):
    """slide -> {'final': {col: val}, 'premicro': {col: val}} of intrinsic TRE, either method's shape.

    VALIS emits per-patient ``*_summary[.csv]`` rows (and a ``_premicro`` variant) keyed by the
    ``from``/``filename`` column: read unchanged into ``final``/``premicro``.

    STARE (tiled) emits one per-slide ``*_tre.json``, keyed by its ``moving`` field. Per
    ``bin/utils/tre_report.py``'s module docstring, ``coarse_tre_px`` is the rigid-stage TRE
    ("directly comparable to VALIS's rigid error") and ``residual_after_px`` is the
    post-registration number ("the analogue of VALIS's non-rigid error"); its p50 is used so it
    lines up with the seg-QC side's ``displacement_*_p50``. STARE has no separate pre-micro stage,
    so both values land in ``final`` — the existing non-rigid fallback in ``reconcile_rows``
    (final's ``non_rigid_D`` when no premicro summary exists) already covers it. Both formats
    write into the same slide dict so a directory containing both is a merge, not a shadow.
    """
    out = {}
    for csv_path in list_files(tre_dir, "*.csv"):
        which = (
            "premicro" if csv_path.name.endswith("_summary_premicro.csv") else "final"
        )
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    slide = row.get("from") or row.get("filename")
                    if not slide:
                        continue
                    slot = out.setdefault(str(slide), {"final": {}, "premicro": {}})[
                        which
                    ]
                    for col in ("rigid_D", "non_rigid_D"):
                        if col in row:
                            slot[col] = _to_float(row[col])
        except (OSError, csv.Error):
            continue
    for json_path in list_files(tre_dir, "*_tre.json"):
        try:
            info = parse_tiled_tre_json(json_path)
        except (OSError, ValueError):
            continue
        slide = info["moving"]
        slot = out.setdefault(str(slide), {"final": {}, "premicro": {}})["final"]
        slot["rigid_D"] = _to_float(info["coarse_tre_px"])
        slot["non_rigid_D"] = _to_float(info["final_p50"])
    return out


def _read_seg_cell_disp(seg_qc_dir):
    """slide -> {stage: (displacement_p50, unit)} from the WARP_SEG_QC JSONs.

    ``unit`` is ``"um"`` when the JSON's own micron column (``displacement_um_p50``)
    is present, else ``"px"`` when it falls back to ``displacement_px_p50``. STARE
    (tiled) runs are the COMMON case for px, not an edge case:
    ``modules/local/warp_seg_qc.nf`` passes no pixel size on that path,
    ``bin/warp_seg_qc.py`` forwards ``None``, and ``bin/utils/cell_pairs.py`` only
    emits ``displacement_um_*`` when a pixel size was given. Callers must never
    average a px value into a µm distribution (or vice versa) — carrying the unit
    alongside the value is what lets them group by it instead of assuming one.
    """
    out = {}
    for jp in list_files(seg_qc_dir, "*.json"):
        try:
            with open(jp, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        slide = data.get("moving")
        stages = data.get("stages") or {}
        if not slide or not isinstance(stages, dict):
            continue
        per_stage = {}
        for stage, metrics in stages.items():
            if not isinstance(metrics, dict):
                continue
            val = metrics.get("displacement_um_p50")
            unit = "um"
            if val is None:
                val = metrics.get(
                    "displacement_px_p50"
                )  # fall back to px if no pixel size
                unit = "px"
            per_stage[stage] = (_to_float(val), unit)
        out[str(slide)] = per_stage
    return out


def _read_seg_qc_match_counts(seg_qc_dir):
    """slide -> matching.n_pairs from each WARP_SEG_QC JSON, when the block is present.

    ``n_pairs`` is how many cells the fixed cell-pairing actually matched (see
    docs/registration_qc.md's ``matching`` block). A displacement p50 computed
    over 12 matched cells is a different claim from one computed over 40 000 --
    read separately from ``_read_seg_cell_disp`` so a caption can say which one a
    run was, without folding a non-stage key into the per-stage metrics dict that
    drives stage discovery.
    """
    out = {}
    for jp in list_files(seg_qc_dir, "*.json"):
        try:
            with open(jp, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        slide = data.get("moving")
        matching = data.get("matching")
        if not slide or not isinstance(matching, dict):
            continue
        n = _to_float(matching.get("n_pairs"))
        if n is not None:
            out[str(slide)] = int(n)
    return out


def reconcile_rows(tre_dir, seg_qc_dir):
    """Pair the registration method's intrinsic TRE with WARP_SEG_QC cell displacement, per slide per stage.

    Returns a list of dicts: ``slide``, ``stage``, ``feature_tre_um``, ``cell_disp_um``,
    ``divergent`` (True/False, or None when either number is missing so no verdict is possible).
    Slides are keyed by slide name (VALIS's ``from``/``filename`` or STARE's ``moving`` — both
    resolve to the same identifier); rows are only emitted for slides that appear in the seg-QC
    output.
    """
    tre = _read_intrinsic_tre(tre_dir)
    cell = _read_seg_cell_disp(seg_qc_dir)
    rows = []
    for slide in sorted(cell):
        slide_tre = tre.get(slide, {})
        for stage in ("rigid", "non_rigid", "micro"):
            col, which = _RECONCILE_TRE_SOURCE[stage]
            feature = slide_tre.get(which, {}).get(col)
            # non_rigid_D is overwritten in place by register_micro(), so only the pre-micro
            # summary isolates the non-rigid TRE. But when no pre-micro summary exists
            # (micro_reg < 2 -> register_micro never ran), the final summary still IS the
            # pre-micro non-rigid value: fall back to it. Without this the non_rigid row is
            # permanently n/a at the shipped default (micro_reg=0), defeating the whole
            # reconciliation for the configuration that actually runs.
            if feature is None and stage == "non_rigid" and which == "premicro":
                feature = slide_tre.get("final", {}).get(col)
            # _read_seg_cell_disp now carries the unit alongside the value (fix round 1,
            # Important 1); this reconciliation only ever compared the numbers, so unpack
            # and keep going -- the unit itself is not reconcile_rows' concern.
            disp_entry = cell.get(slide, {}).get(stage)
            disp = disp_entry[0] if disp_entry is not None else None
            divergent = None
            if feature is not None and disp is not None:
                lo, hi = sorted((abs(feature), abs(disp)))
                divergent = hi > RECONCILE_DIVERGENCE_RATIO * lo if lo > 0 else hi > 0
            rows.append(
                {
                    "slide": slide,
                    "stage": stage,
                    "feature_tre_um": feature,
                    "cell_disp_um": disp,
                    "divergent": divergent,
                }
            )
    return rows


SCATTER_W = 460
SCATTER_H = 360


def _reconciliation_scatter_svg(points, ratio_band):
    """Feature-TRE (x) against cell displacement (y), log-log, with the divergence band.

    ``points`` is ``[(feature_tre, cell_disp, label), ...]``, one entry per
    slide-stage. ``reconcile_rows`` calls a row divergent when the larger of the
    two numbers exceeds ``ratio_band`` times the smaller — a symmetric rule, so
    the region it accepts is exactly the strip between ``y = x·r`` and
    ``y = x/r``. On log axes those are two straight lines parallel to the
    identity, which is why the band can be DRAWN instead of recomputed per point:
    a reader sees which slide-stage is out and by how far, rather than a column
    of ticks and crosses.

    Log axes because the quantity being judged is a ratio, and because the stages
    span two orders of magnitude (rigid ≈ 6 µm, micro ≈ 0.8 µm) on one plot. The
    cost is that a non-positive or missing coordinate cannot be placed at all;
    those are counted in the caption rather than disappearing.
    """
    usable = []
    dropped = 0
    for x, y, label in points:
        vx, vy = _finite([x]), _finite([y])
        if vx and vy and vx[0] > 0 and vy[0] > 0:
            usable.append((math.log10(vx[0]), math.log10(vy[0]), vx[0], vy[0], label))
        else:
            dropped += 1
    if not usable:
        return (
            '<p class="empty-notice">No slide-stage has both a feature-TRE and a '
            "cell displacement that can be placed on a log axis.</p>"
        )
    lo = min(min(p[0], p[1]) for p in usable) - 0.3
    hi = max(max(p[0], p[1]) for p in usable) + 0.3
    if hi - lo < 0.6:
        hi = lo + 0.6
    pad_l, pad_b, pad_t, pad_r = 58, 44, 26, 14
    pw = SCATTER_W - pad_l - pad_r
    ph = SCATTER_H - pad_t - pad_b

    def sx(v):
        return pad_l + pw * (v - lo) / (hi - lo)

    def sy(v):
        return pad_t + ph - ph * (v - lo) / (hi - lo)

    d = math.log10(ratio_band)
    parts = [
        f'<svg width="{SCATTER_W}" height="{SCATTER_H}" role="img" '
        f'aria-label="feature-TRE versus cell displacement" '
        f'style="border:1px solid #eee;">',
        '<clipPath id="reconcile-clip">'
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}"/></clipPath>',
        # NOT an f-string: ruff's F541 (pyflakes, in this repo's `select`) rejects
        # an f-prefix with no placeholder, and this line has none.
        '<g clip-path="url(#reconcile-clip)">',
        # the shaded agreement strip, then its two edges
        f'<polygon points="{sx(lo):.1f},{sy(lo + d):.1f} {sx(hi):.1f},{sy(hi + d):.1f} '
        f'{sx(hi):.1f},{sy(hi - d):.1f} {sx(lo):.1f},{sy(lo - d):.1f}" '
        'fill="#27ae60" fill-opacity="0.08"/>',
        f'<line class="identity" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" '
        f'x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" stroke="#999" stroke-dasharray="2,3"/>',
        f'<line class="band" x1="{sx(lo):.1f}" y1="{sy(lo + d):.1f}" '
        f'x2="{sx(hi):.1f}" y2="{sy(hi + d):.1f}" stroke="#27ae60" '
        'stroke-dasharray="5,3"/>',
        f'<line class="band" x1="{sx(lo):.1f}" y1="{sy(lo - d):.1f}" '
        f'x2="{sx(hi):.1f}" y2="{sy(hi - d):.1f}" stroke="#27ae60" '
        'stroke-dasharray="5,3"/>',
    ]
    for lx, ly, vx, vy, label in usable:
        small, large = sorted((vx, vy))
        divergent = large > ratio_band * small
        parts.append(
            f'<circle cx="{sx(lx):.1f}" cy="{sy(ly):.1f}" r="4.5" '
            f'fill="{"#c0392b" if divergent else "#27ae60"}" fill-opacity="0.75">'
            f"<title>{html.escape(str(label))}: feature-TRE {vx:.3g}, "
            f"cell disp {vy:.3g}{' — DIVERGENT' if divergent else ''}</title>"
            "</circle>"
        )
    parts.append("</g>")
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" y2="{pad_t + ph}" '
        'stroke="#999"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#999"/>'
    )
    parts.append(
        f'<text x="{pad_l + pw / 2:.0f}" y="{SCATTER_H - 6}" font-size="10" '
        'fill="#444" text-anchor="middle">feature-TRE (µm, log scale)</text>'
    )
    parts.append(
        f'<text x="12" y="{pad_t + ph / 2:.0f}" font-size="10" fill="#444" '
        f'transform="rotate(-90 12 {pad_t + ph / 2:.0f})" text-anchor="middle">'
        "cell displacement p50 (µm, log scale)</text>"
    )
    parts.append(
        f'<text x="{pad_l}" y="{pad_t - 10}" font-size="10" fill="#666">'
        f"shaded: the two measures agree within {ratio_band:g}×"
        + (f"; {dropped} point(s) not plottable" if dropped else "")
        + "</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def reconciliation_section(tre_dir, seg_qc_dir):
    """Render the feature-TRE vs cell-displacement reconciliation as a log-log scatter."""
    title = "Feature-TRE vs Cell-Displacement Reconciliation"
    rows = reconcile_rows(tre_dir, seg_qc_dir)
    if not rows:
        return section(
            title,
            '<p class="empty-notice">No overlapping registration TRE + segmentation-warp QC '
            "found to reconcile.</p>",
        )

    points = [
        (r["feature_tre_um"], r["cell_disp_um"], f"{r['slide']} · {r['stage']}")
        for r in rows
        if r["divergent"] is not None
    ]
    n_divergent = sum(1 for r in rows if r["divergent"])
    n_no_verdict = sum(1 for r in rows if r["divergent"] is None)
    caption = (
        "<p style='font-size:0.85rem;color:#666;margin:0 0 8px;'>"
        "The registration method's feature-TRE is measured on its own keypoints "
        "(optimistic); cell displacement is measured on independently segmented "
        "nuclei. Each point is one slide-stage. A point outside the shaded band "
        f"disagrees by more than {RECONCILE_DIVERGENCE_RATIO:g}× — worth a look. "
        f"{n_divergent} of {len(points)} comparable point(s) diverge"
        + (
            f"; {n_no_verdict} slide-stage row(s) had no comparable pair."
            if n_no_verdict
            else "."
        )
        + "</p>"
    )
    return section(
        title,
        caption
        + "\n"
        + _reconciliation_scatter_svg(points, RECONCILE_DIVERGENCE_RATIO),
    )


def _read_residual_column(csv_path):
    """Stream a ``*_reg_residuals.csv`` into ``(values, stage_label, n_rows)``.

    bin/warp_seg_qc.py's write_per_cell_csv writes one row per matched cell pair,
    uncapped — a real slide is 10^4-10^6 rows — with columns
    ``moving,ref_x,ref_y,residual_px,stage``. ``csv.DictReader`` still builds one
    dict per row internally, but nothing from it survives past that iteration
    except the numeric residual, so what this function RETAINS is a list of
    floats, not a list of per-row dicts. ``n_rows`` counts EVERY data row,
    including one whose residual could not be parsed OR was non-finite (``inf``/
    ``nan`` — ``_to_float`` only drops NaN, not infinity), so the plot's n and
    the "(N unparseable)" count in ``seg_residuals_section`` stay consistent with
    each other: a value this function keeps is a value ``_finite`` will plot.
    """
    values, stages, n_rows = [], [], 0
    with open(csv_path, newline="") as fh:
        # No islice/limit here, deliberately: the histogram needs every row.
        # The function this replaced sliced the head and counted the tail, and
        # the slice had to be itertools.islice -- zip(reader, range(limit))
        # pulls one item it never yields, so the tail count came out one short
        # (measured 399 999 on a 400 000-row file). Nothing here can make that
        # mistake because nothing here has a limit.
        for row in csv.DictReader(fh):
            n_rows += 1
            v = _to_float(row.get("residual_px"))
            if v is not None and math.isfinite(v):
                values.append(v)
            stage = row.get("stage")
            if stage and stage not in stages:
                stages.append(stage)
    return values, "/".join(stages), n_rows


def seg_residuals_section(seg_residuals_dir):
    """Per-cell registration residuals as one error distribution per slide CSV.

    This replaced a 500-row head-of-CSV table (spec Phase 4). Five hundred rows out
    of a possible million answered no question a reader had; the distribution of all
    of them, with p50 and p90 marked, answers the one they did. The CSVs themselves
    are published unchanged into the report's ``seg_residuals/`` data folder and are
    joined onto cell labels by the SpatialData export, so nothing is lost.
    """
    csvs = list_files(seg_residuals_dir, "*.csv")
    if not csvs:
        return section(
            "Per-Cell Registration Residuals",
            '<p class="empty-notice">No per-cell registration residuals found.</p>',
        )
    parts = ['<div style="display:flex;flex-wrap:wrap;gap:12px;">']
    for csv_path in csvs:
        name = Path(csv_path).name
        try:
            values, stage, n_rows = _read_residual_column(csv_path)
        except Exception as exc:  # noqa: BLE001 - report, never crash
            parts.append(
                '<p class="empty-notice">Could not parse '
                f"{html.escape(name)}: {html.escape(str(exc))}</p>"
            )
            continue
        label = f"{name} — {stage} stage" if stage else name
        if n_rows != len(values):
            label += f" ({n_rows - len(values)} unparseable)"
        parts.append(_error_distribution_svg(values, title=label, bins=30))
    parts.append("</div>")
    parts.append(
        "<p style='font-size:0.8rem;color:#888;margin-top:6px;'>"
        "The full per-cell CSVs are published unchanged under "
        "<code>seg_residuals/</code> in this report's data folder.</p>"
    )
    return section("Per-Cell Registration Residuals", "\n".join(parts))


def seg_overlay_section(postprocess_dir):
    """Dedicated section for the *_seg_overlay* PNGs (the most-inspected QC)."""
    all_pngs = list_files(postprocess_dir, "*.png")
    overlays = [p for p in all_pngs if "_seg_overlay" in p.name]
    return section("Segmentation Overlays", img_grid(overlays, wide=True))


def copy_data(args):
    """Copy the raw QC artifacts (PNGs, CSVs) into the report's data directory."""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    def copy_glob(src_dir, pattern, subdir):
        """Copy every file matching ``pattern`` in ``src_dir`` into ``data_dir/subdir``."""
        files = list_files(src_dir, pattern)
        if not files:
            return
        dest = data_dir / subdir
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / Path(f).name)

    copy_glob(args.preprocess_qc, "*.png", "preprocess_qc")
    copy_glob(args.registration_qc, "*.png", "registration_qc")
    copy_glob(args.registration_tre, "*.csv", "registration_tre")
    copy_glob(args.registration_tre, "*_tre.json", "registration_tre")
    copy_glob(args.postprocess_qc, "*.png", "postprocess_qc")
    copy_glob(args.seg_qc, "*.json", "seg_qc")
    copy_glob(args.seg_residuals, "*.csv", "seg_residuals")
    if args.run_summary and Path(args.run_summary).exists():
        shutil.copy2(args.run_summary, data_dir / "run_summary.json")
    if args.versions and Path(args.versions).exists():
        shutil.copy2(args.versions, data_dir / "collated_versions.yml")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """CLI entry point: assemble the per-step QC outputs into a single HTML report."""
    configure_logging(level=logging.INFO)
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    summary = parse_run_summary_json(args.run_summary)
    present = {
        "Preprocessing": bool(list_files(args.preprocess_qc, "*.png")),
        "Registration": bool(list_files(args.registration_qc, "*")),
        "Segmentation & Quant": bool(list_files(args.postprocess_qc, "*.png")),
    }

    html_parts = [html_header(timestamp)]
    html_parts.append(run_summary_section(summary))
    html_parts.append(status_strip_section(present))
    html_parts.append(preprocess_qc_section(args.preprocess_qc))
    html_parts.append(
        registration_qc_section(
            args.registration_qc,
            args.registration_tre,
            args.seg_qc,
        )
    )
    html_parts.append(reconciliation_section(args.registration_tre, args.seg_qc))
    html_parts.append(seg_residuals_section(args.seg_residuals))
    html_parts.append(seg_overlay_section(args.postprocess_qc))
    html_parts.append(postprocess_qc_section(args.postprocess_qc))
    html_parts.append(html_footer())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(html_parts))

    logger.info(f"Report written to: {out_path}")

    copy_data(args)
    logger.info(f"Data copied to: {args.data_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
