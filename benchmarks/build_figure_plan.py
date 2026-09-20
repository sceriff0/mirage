#!/usr/bin/env python3
"""build_figure_plan.py -- expand configs/figures.yaml into the rows submit_figures.sh runs.

One row per FIGURE, TSV, kind first, so the shell reads them with `while IFS=$'\\t' read`:

    mosaic   <patch_um> <variants>
    overlay  <arm>     <field_um> <zoom_um> <variants>
    zoom     <method>  <field_um> <mask>    <crop>
    crop     <method>  <field_um> <mask>    <crop_px>
    channel  <arm>     <field_um> <crop_px> <channel> <color>

Ordered by what they depend on -- mosaic (every arm), overlays (one arm each), then the
segmentation figures method by method, then the channel crops -- so the launcher finishes a
run before drawing from it and never revisits one.

The expensive axes (`arms`, `segmentation.methods`) are validated hard, here at plan time
rather than hours into a job: an unknown arm name or a seg_method the pipeline does not
accept fails immediately. The size axes are crossed freely -- each is a re-render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEG_METHODS = ("stardist", "instantseg", "cellsam")
MASKS = ("cell", "nuclei", "both")
CROP_MODES = ("none", "also", "only")
# the arms benchmarks/submit_mosaic.sh knows how to build; `ashlar` expands to its
# tile/shift-derived directory name, which only the launcher can spell
ARMS = ("valis_high_micro2", "stare_high", "ashlar")


def _schema_seg_methods() -> tuple[str, ...]:
    """The enum nextflow_schema.json actually declares, so this file cannot drift from it."""
    schema = REPO / "nextflow_schema.json"
    if not schema.is_file():
        return SEG_METHODS

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("seg_method"), dict):
                enum = node["seg_method"].get("enum")
                if enum:
                    return tuple(enum)
            for value in node.values():
                found = walk(value)
                if found:
                    return found
        return None

    return walk(json.loads(schema.read_text())) or SEG_METHODS


def load(path: Path) -> dict:
    import yaml

    if not path.is_file():
        raise SystemExit(f"{path}: not found")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level")
    return cfg


def _numbers(section: dict, key: str, where: str, allow_zero=False) -> list[float]:
    values = section.get(key) or []
    if not isinstance(values, list) or not values:
        raise SystemExit(f"{where}.{key}: expected a non-empty list")
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise SystemExit(f"{where}.{key}: {v!r} is not a number") from None
        if f < 0 or (f == 0 and not allow_zero):
            raise SystemExit(f"{where}.{key}: {v!r} must be positive")
        out.append(f)
    return out


def _count(section: dict, key: str, where: str, default=1) -> int:
    value = section.get(key, default)
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"{where}.{key}: {value!r} is not a whole number") from None
    if n < 1:
        raise SystemExit(f"{where}.{key}: must be >= 1, got {n}")
    return n


def plan(cfg: dict) -> list[tuple]:
    arms = cfg.get("arms") or []
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"arms: {bad} unknown; submit_figures.sh builds {list(ARMS)}")
    if len(set(arms)) != len(arms):
        raise SystemExit(f"arms: repeated entries in {arms}")

    seg = cfg.get("segmentation") or {}
    methods = seg.get("methods") or []
    allowed = _schema_seg_methods()
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise SystemExit(
            f"segmentation.methods: {bad} not accepted by the pipeline; seg_method is one "
            f"of {list(allowed)}"
        )
    if len(set(methods)) != len(methods):
        raise SystemExit(f"segmentation.methods: repeated entries in {methods}")

    ref = cfg.get("reference_arm") or (arms[0] if arms else None)
    if (methods or (cfg.get("figures") or {}).get("channels")) and not ref:
        raise SystemExit(
            "reference_arm: needed -- segmentation and the channel crops resume from one arm"
        )
    if ref and arms and ref not in arms:
        raise SystemExit(f"reference_arm: {ref!r} is not in arms {arms}")

    figures = cfg.get("figures") or {}
    rows: list[tuple] = []

    mosaic = figures.get("mosaic")
    if mosaic:
        if len(arms) < 1:
            raise SystemExit("figures.mosaic: needs at least one arm")
        variants = _count(mosaic, "variants", "figures.mosaic")
        for patch in _numbers(mosaic, "patch_um", "figures.mosaic"):
            rows.append(("mosaic", patch, variants))

    overlay = figures.get("overlay")
    if overlay:
        over_arms = overlay.get("arms") or arms
        bad = [a for a in over_arms if a not in arms]
        if bad:
            raise SystemExit(f"figures.overlay.arms: {bad} not in arms {arms}")
        variants = _count(overlay, "variants", "figures.overlay")
        fields = _numbers(overlay, "field_um", "figures.overlay")
        # zoom_um is optional: absent means one overlay per field, with no inset
        zooms = (
            _numbers(overlay, "zoom_um", "figures.overlay", allow_zero=True)
            if overlay.get("zoom_um")
            else [0.0]
        )
        for arm in over_arms:
            for field in fields:
                for zoom in zooms:
                    rows.append(("overlay", arm, field, zoom, variants))

    zoom_cfg = figures.get("zoom") or {}
    crop_cfg = figures.get("crop") or {}
    zoom_fields = _numbers(zoom_cfg, "field_um", "figures.zoom") if zoom_cfg else []
    masks = zoom_cfg.get("masks") or ["cell"]
    bad = [m for m in masks if m not in MASKS]
    if bad:
        raise SystemExit(f"figures.zoom.masks: {bad} not in {list(MASKS)}")
    crop_mode = zoom_cfg.get("crop", "none")
    if crop_mode not in CROP_MODES:
        raise SystemExit(f"figures.zoom.crop: {crop_mode!r} not in {list(CROP_MODES)}")
    crop_sizes = (
        [int(v) for v in _numbers(crop_cfg, "crop_px", "figures.crop")]
        if crop_cfg
        else []
    )
    if zoom_cfg and not methods:
        raise SystemExit(
            "figures.zoom: needs segmentation.methods -- it draws segmented cells"
        )

    for method in methods:
        for field in zoom_fields:
            for mask in masks:
                rows.append(("zoom", method, field, mask, crop_mode))
                for crop_px in crop_sizes:
                    rows.append(("crop", method, field, mask, crop_px))

    channels = figures.get("channels") or {}
    names = channels.get("names") or []
    if names:
        if not all(isinstance(c, str) and c.strip() for c in names):
            raise SystemExit(f"figures.channels.names: expected names, got {names}")
        colors = channels.get("colors") or ["white"]
        fields = _numbers(channels, "field_um", "figures.channels")
        sizes = [int(v) for v in _numbers(channels, "crop_px", "figures.channels")]
        for i, name in enumerate(names):
            for field in fields:
                for crop_px in sizes:
                    rows.append(
                        ("channel", ref, field, crop_px, name, colors[i % len(colors)])
                    )
    return rows


def format_rows(rows) -> str:
    def cell(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    return "\n".join("\t".join(cell(v) for v in row) for row in rows)


def summary(cfg: dict, rows) -> str:
    kinds = {}
    for row in rows:
        kinds[row[0]] = kinds.get(row[0], 0) + 1
    methods = len((cfg.get("segmentation") or {}).get("methods") or [])
    arms = len(cfg.get("arms") or [])
    drawn = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "nothing"
    return (
        f"{arms} registration arm(s) + {methods} segmentation run(s) "
        f"-> {drawn} ({len(rows)} figure job(s))"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=REPO / "benchmarks" / "configs" / "figures.yaml"
    )
    ap.add_argument(
        "--count", action="store_true", help="print what it expands to, not the plan"
    )
    args = ap.parse_args(argv)
    cfg = load(args.config)
    rows = plan(cfg)
    print(summary(cfg, rows) if args.count else format_rows(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
