#!/usr/bin/env python3
"""build_figure_plan.py -- expand configs/figures.yaml into the rows submit_figures.sh runs.

One row per FIGURE, TSV, four fields:

    <kind>  <run key>  <output subdirectory>  <shell-quoted arguments>

The launcher resolves the run key (``arm:<label>`` -> that arm's directory, ``seg:<method>``
-> its segmentation run, ``arms:all`` -> every arm, for the mosaic), makes the directory and
runs the tool with the arguments verbatim. Everything a figure needs is therefore decided
HERE, in Python, where it can be tested; the shell decides nothing. Rows are ordered by what
they depend on -- mosaic and overlays off the registration arms, then the segmentation
figures method by method, then the channel crops -- so a run is finished before it is read.

THE COST MODEL, which is what the shape of this file is for. Only two things are expensive:
one REGISTRATION per arm and one SEGMENTATION per method. Every other axis -- field size,
patch size, output size, mask, ROI, overlay/checker, contrast mode, variant, channel, patient
-- is a re-render of slides already on disk, so they are crossed freely while the two
expensive axes are validated hard, here, in seconds: an unknown arm, or a seg_method the
schema does not declare, fails before a single node-hour is spent.

An arm given as ``{name: <label>, dir: <path>}`` ALREADY EXISTS -- a finished run of the arms
benchmark, say. It is never rebuilt, and if it ran at reg_qc=2 it already carries the
*_seg_qc.json the mosaic prints its Dice from, so no registration is repeated to get numbers.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEG_METHODS = ("stardist", "instantseg", "cellsam")
MASKS = ("cell", "nuclei", "both")
CROP_MODES = ("none", "also", "only")
NUMBERS = ("auto", "scorer", "image", "none")
KINDS = ("overlay", "checker")
AUTOSCALE = ("clean", "percentile")
# an unlabelled figure is worse than an honestly labelled one: a title with nothing behind it
# reads as an oversight, "NA" reads as a fact
NA = "NA"
# the arms submit_figures.sh knows how to BUILD; anything else must name its own dir
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


# --- small validators -------------------------------------------------------------
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


def _choices(section: dict, key: str, where: str, allowed, default) -> list[str]:
    values = section.get(key) or default
    if not isinstance(values, list):
        values = [values]
    bad = [v for v in values if v not in allowed]
    if bad:
        raise SystemExit(f"{where}.{key}: {bad} not in {list(allowed)}")
    return list(values)


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def title_of(label: str | None) -> str:
    """What a figure writes top left. An arm or method with no name prints NA, not nothing."""
    return str(label).strip() if label and str(label).strip() else NA


# --- the axes every figure shares --------------------------------------------------
def rois(cfg: dict) -> list[tuple[str, str]]:
    """[(directory suffix, "Y,X"), ...] -- one per region, or a single empty one for `auto`.

    Every registered slide of a patient is on the same reference canvas, so one ROI names the
    same tissue in every arm, method, size and channel. Several ROIs multiply every figure
    below at no cost and keep that comparability within each region.
    """
    value = (cfg.get("options") or {}).get("roi") or ""
    values = value if isinstance(value, list) else [value]
    values = [str(v).strip() for v in values if str(v).strip()]
    if not values:
        return [("", "")]
    for v in values:
        parts = v.split(",")
        if len(parts) != 2 or not all(p.strip().lstrip("-").isdigit() for p in parts):
            raise SystemExit(
                f'options.roi: {v!r} is not "Y,X" in full-resolution pixels'
            )
    if len(values) == 1:
        return [("", values[0])]
    return [(f"_r{i}", v) for i, v in enumerate(values, 1)]


def patients(cfg: dict) -> list[str]:
    """The patients to draw, or [""] meaning 'whatever the run holds'."""
    value = (cfg.get("options") or {}).get("patient") or ""
    values = value if isinstance(value, list) else [value]
    return [str(v).strip() for v in values if str(v).strip()] or [""]


def arm_names(cfg: dict) -> list[str]:
    return [e["name"] if isinstance(e, dict) else e for e in cfg.get("arms") or []]


def arm_dirs(cfg: dict) -> dict[str, str]:
    """{label: directory} for arms that already exist and must never be rebuilt."""
    return {
        e["name"]: str(e["dir"])
        for e in cfg.get("arms") or []
        if isinstance(e, dict) and e.get("dir")
    }


def _validate_arms(cfg: dict) -> tuple[list[str], dict[str, str]]:
    for entry in cfg.get("arms") or []:
        if isinstance(entry, dict):
            if not entry.get("name"):
                raise SystemExit(f"arms: {entry} has no name")
            if not entry.get("dir"):
                raise SystemExit(
                    f"arms: {entry['name']!r} is a mapping, so it must carry `dir:` -- the "
                    "run to draw from. Use a plain name to have this grid build the arm."
                )
    arms, external = arm_names(cfg), arm_dirs(cfg)
    bad = [a for a in arms if a not in ARMS and a not in external]
    if bad:
        raise SystemExit(
            f"arms: {bad} unknown; submit_figures.sh builds {list(ARMS)}, or give "
            "{name: <label>, dir: <an existing run>} to draw from one you already have"
        )
    if len(set(arms)) != len(arms):
        raise SystemExit(f"arms: repeated entries in {arms}")
    return arms, external


def _validate_methods(cfg: dict) -> list[str]:
    methods = (cfg.get("segmentation") or {}).get("methods") or []
    allowed = _schema_seg_methods()
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise SystemExit(
            f"segmentation.methods: {bad} not accepted by the pipeline; seg_method is one "
            f"of {list(allowed)}"
        )
    if len(set(methods)) != len(methods):
        raise SystemExit(f"segmentation.methods: repeated entries in {methods}")
    return methods


# --- the plan ----------------------------------------------------------------------
def plan(cfg: dict) -> list[tuple[str, str, str, str]]:
    arms, external = _validate_arms(cfg)
    methods = _validate_methods(cfg)
    figures = cfg.get("figures") or {}
    regions, pids = rois(cfg), patients(cfg)

    ref = cfg.get("reference_arm") or (arms[0] if arms else None)
    if (methods or figures.get("channels")) and not ref:
        raise SystemExit(
            "reference_arm: needed -- segmentation and the channel crops resume from one arm"
        )
    if ref and arms and ref not in arms:
        raise SystemExit(f"reference_arm: {ref!r} is not in arms {arms}")

    opt = cfg.get("options") or {}
    outline = [
        "--outline-color",
        str(opt.get("outline_color", "#ffd400")),
        "--nuclei-color",
        str(opt.get("nuclei_color", "#00e5ff")),
        "--outline-width",
        str(opt.get("outline_width", 1)),
        "--channel-label",
        str(opt.get("channel_label", "DAPI")),
    ]
    contrast = ["--sat", str(opt.get("sat", 0.35)), "--bg-k", str(opt.get("bg_k", 3.0))]

    rows: list[tuple[str, str, str, str]] = []

    def add(kind, run, outdir, args):
        rows.append((kind, run, outdir, shlex.join(str(a) for a in args)))

    mosaic = figures.get("mosaic")
    if mosaic:
        if not arms:
            raise SystemExit("figures.mosaic: needs at least one arm")
        variants = _count(mosaic, "variants", "figures.mosaic")
        kinds = _choices(mosaic, "kinds", "figures.mosaic", KINDS, ["overlay"])
        numbers = _choices(mosaic, "numbers", "figures.mosaic", NUMBERS, ["auto"])
        for patch in _numbers(mosaic, "patch_um", "figures.mosaic"):
            for kind in kinds:
                for number in numbers:
                    for suffix, roi in regions:
                        args = ["--patch-um", _num(patch), "--variants", variants]
                        args += ["--kinds", kind, "--numbers", number]
                        if roi:
                            args += ["--roi", roi]
                        for pid in pids:  # reg_mosaic takes --patient repeatedly
                            if pid:
                                args += ["--patient", pid]
                        add(
                            "mosaic",
                            "arms:all",
                            f"mosaic/p{_num(patch)}_{kind}_{number}{suffix}",
                            args,
                        )

    overlay = figures.get("overlay")
    if overlay:
        over_arms = overlay.get("arms") or arms
        bad = [a for a in over_arms if a not in arms]
        if bad:
            raise SystemExit(f"figures.overlay.arms: {bad} not in arms {arms}")
        variants = _count(overlay, "variants", "figures.overlay")
        fields = _numbers(overlay, "field_um", "figures.overlay")
        zooms = (
            _numbers(overlay, "zoom_um", "figures.overlay", allow_zero=True)
            if overlay.get("zoom_um")
            else [0.0]
        )
        for arm in over_arms:
            for field in fields:
                for zoom in zooms:
                    for suffix, roi in regions:
                        for pid in pids:
                            args = ["--field-um", _num(field), "--variants", variants]
                            args += ["--title", title_of(arm)]
                            if zoom:
                                args += ["--zoom-um", _num(zoom)]
                            if roi:
                                args += ["--roi", roi]
                            if pid:
                                args += ["--patient", pid]
                            add(
                                "overlay",
                                f"arm:{arm}",
                                f"overlay/{arm}/f{_num(field)}_z{_num(zoom)}{suffix}",
                                args,
                            )

    zoom_cfg = figures.get("zoom") or {}
    crop_cfg = figures.get("crop") or {}
    if zoom_cfg and not methods:
        raise SystemExit(
            "figures.zoom: needs segmentation.methods -- it draws segmented cells"
        )
    zoom_fields = _numbers(zoom_cfg, "field_um", "figures.zoom") if zoom_cfg else []
    masks = _choices(zoom_cfg, "masks", "figures.zoom", MASKS, ["cell"])
    crop_mode = zoom_cfg.get("crop", "none")
    if crop_mode not in CROP_MODES:
        raise SystemExit(f"figures.zoom.crop: {crop_mode!r} not in {list(CROP_MODES)}")
    crop_sizes = (
        [int(v) for v in _numbers(crop_cfg, "crop_px", "figures.crop")]
        if crop_cfg
        else []
    )
    for method in methods:
        for field in zoom_fields:
            for mask in masks:
                for suffix, roi in regions:
                    for pid in pids:
                        base = ["--mask", mask, "--field-um", _num(field)]
                        base += ["--title", title_of(method), *outline]
                        if roi:
                            base += ["--roi", roi]
                        if pid:
                            base += ["--patient", pid]
                        add(
                            "zoom",
                            f"seg:{method}",
                            f"zoom/{method}/f{_num(field)}_{mask}{suffix}",
                            base + ["--crop", crop_mode],
                        )
                        for crop_px in crop_sizes:
                            add(
                                "crop",
                                f"seg:{method}",
                                f"crops/{method}/f{_num(field)}_p{crop_px}_{mask}{suffix}",
                                base + ["--crop", "only", "--crop-px", crop_px],
                            )

    channels = figures.get("channels") or {}
    names = channels.get("names") or []
    if names:
        if not all(isinstance(c, str) and c.strip() for c in names):
            raise SystemExit(f"figures.channels.names: expected names, got {names}")
        colors = channels.get("colors") or ["white"]
        fields = _numbers(channels, "field_um", "figures.channels")
        sizes = [int(v) for v in _numbers(channels, "crop_px", "figures.channels")]
        scales = _choices(
            channels, "autoscale", "figures.channels", AUTOSCALE, ["clean"]
        )
        for i, name in enumerate(names):
            for field in fields:
                for crop_px in sizes:
                    for scale in scales:
                        for suffix, roi in regions:
                            for pid in pids:
                                args = ["--channel", name]
                                args += ["--colors", colors[i % len(colors)]]
                                args += ["--field-um", _num(field)]
                                args += ["--crop-px", crop_px, "--autoscale", scale]
                                args += ["--title", title_of(ref), *contrast]
                                if roi:
                                    args += ["--roi", roi]
                                if pid:
                                    args += ["--patient", pid]
                                add(
                                    "channel",
                                    f"arm:{ref}",
                                    f"crops/channels/f{_num(field)}_p{crop_px}_{scale}{suffix}",
                                    args,
                                )
    return rows


def format_rows(rows) -> str:
    return "\n".join("\t".join(row) for row in rows)


def summary(cfg: dict, rows) -> str:
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row[0]] = kinds.get(row[0], 0) + 1
    external = arm_dirs(cfg)
    build = [a for a in arm_names(cfg) if a not in external]
    methods = len(_validate_methods(cfg))
    drawn = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "nothing"
    return (
        f"{len(build)} registration arm(s) to build, {len(external)} reused, "
        f"{methods} segmentation run(s) -> {drawn} ({len(rows)} figure job(s))"
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
