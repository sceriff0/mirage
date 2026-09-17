#!/usr/bin/env python3
"""reg_overlay.py -- DAPI Before and After images of one registration, one crop each.

The single-pair companion of reg_mosaic.py. Where the mosaic tiles many small ROIs
across arms, this draws ONE larger field of view twice, from one registration output
directory (a mirage ``--outdir`` that ran registration with reg_qc >= 1):

    <outdir>/<patient>_<round>_before.png/.pdf   moving DAPI as acquired over the reference
    <outdir>/<patient>_<round>_after.png/.pdf    registered moving DAPI over the reference
    <outdir>/<patient>_<round>_locator.png/.pdf  the reference with the crop boxed
    <outdir>/<patient>_<round>_overlay.json      crop (reference frame, full-res px), numbers

Both images are the SAME reference-frame crop of the pipeline's two-panel QC composite
(<slide>_QC_RGB_fullres.tif), so they differ only by the registration. Colours default
to magenta = moving, cyan = reference (overlap white); each image carries a µm scale bar
and ``Dice = X  Δ = Y µm`` -- from the reg_qc=2 scorer when its JSON exists, else computed
from the crop (reg_mosaic.image_metrics; --numbers).

THE CROP. Chosen on the reference DAPI by the same tissue x texture score as the mosaic
(reg_mosaic.select_rois), or given with --roi Y,X. --avoid-rois-json takes a mosaic's
<patient>_rois.json and keeps this crop clear of every mosaic ROI, so the two figures
show different tissue.

    python -m benchmarks.reg_overlay results/valis_high_micro2 --patient 033 \\
        --field-um 500 --avoid-rois-json mosaic/033_rois.json -o figs/overlay
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmarks import reg_mosaic as rm

log = logging.getLogger("reg_overlay")


def draw_panel(
    img, title, note, scalebar, out_stem: Path, formats, dpi, legend, size_in=None
):
    plt = rm._mpl()
    h, w = img.shape[:2]
    # one image pixel per output pixel unless a size is forced: a smaller figure downsamples
    size_in = size_in or w / dpi
    fig, ax = plt.subplots(figsize=(size_in, size_in * h / w))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    font = max(11.0, size_in * 2.2)
    rm._outline(
        ax.text(
            0.03,
            0.97,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=font,
            color="white",
            fontweight="bold",
        )
    )
    if note:
        rm._outline(
            ax.text(
                0.97,
                0.03,
                note,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=font * 0.85,
                color="white",
            )
        )
    if scalebar:
        rm.draw_scalebar(ax, h, w, scalebar[0], scalebar[1], font * 0.85, thick=0.01)
    if legend:
        fig.text(0.0, -0.004, legend, ha="left", va="top", fontsize=font * 0.6)
    for fmt in formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_avoid(path: Path | None, reference: Path) -> list[tuple[int, int, int]]:
    """The mosaic's ROIs as (y, x, size) boxes -- only if it used the SAME reference slide.

    ROI coordinates live in the reference's frame. A mosaic registered onto another
    reference has another frame, where the same numbers name different tissue, so its
    ROIs are ignored (with a warning) rather than applied as if they were comparable.
    """
    if not path:
        return []
    d = json.loads(Path(path).read_text())
    theirs = Path(d.get("reference", "")).name
    if theirs != Path(reference).name:
        log.warning(
            "%s was drawn on reference %s, this run registers onto %s: different frames, "
            "so its ROIs are not avoided",
            path,
            theirs or "?",
            Path(reference).name,
        )
        return []
    size = int(d["patch_px"])
    return [(int(r["y"]), int(r["x"]), size) for r in d["rois"]]


def render(arm: rm.Arm, key: str, opt: argparse.Namespace) -> dict:
    sl = arm.slide(key)
    comp = arm.composite(key)
    if not comp.has_before:
        raise SystemExit(
            f"{comp.src.path.name}: no Before panel (QC ran without --native)"
        )
    H, W = comp.canvas
    px = opt.pixel_size_um or comp.px or arm.px
    if opt.field_px:
        field_px = opt.field_px
    elif px:
        field_px = int(round(opt.field_um / px))
    else:
        raise SystemExit("pixel size unknown; pass --pixel-size-um or --field-px")
    field_px = min(field_px, H, W)

    if opt.roi:
        y, x = (int(v) for v in opt.roi.split(",")[:2])
    else:
        factor_hint = max(1, int(round(opt.lowres_um / px))) if px else 16
        low, f = comp.lowres_reference(factor_hint)
        picks = rm.select_rois(
            low,
            f,
            field_px,
            1,
            0.0,
            0.0,
            (H, W),
            exclude=load_avoid(opt.avoid_rois_json, arm.ref.image),
        )
        if not picks:
            raise SystemExit(
                f"{arm.patient}/{key}: no tissue crop of {field_px} px clear of the avoided "
                "ROIs; lower --field-um or pass --roi"
            )
        y, x = picks[0]
    log.info(
        "%s %s: crop y=%d x=%d size=%d px (%s µm/px)",
        arm.patient,
        key,
        y,
        x,
        field_px,
        px,
    )

    ref_a, mov_a, used_a = arm.crop(key, "after", y, x, field_px, field_px)
    ref_b, mov_b, used_b = arm.crop(key, "before", y, x, field_px, field_px)
    # one stretch for the reference (it is the same pixels in both panels); the moving
    # channel is stretched per panel, since the two crops cover different tissue
    lim_ref = rm.percentile_limits(ref_a, opt.pmin, opt.pmax)
    panels = {}
    qc = (
        None
        if opt.numbers == "image"
        else arm.seg_qc(key, warn=opt.numbers == "scorer")
    )
    for name, (ref, mov) in {"before": (ref_b, mov_b), "after": (ref_a, mov_a)}.items():
        lim_mov = rm.percentile_limits(mov, opt.pmin, opt.pmax)
        img = rm.overlay(
            rm.stretch(mov, lim_mov, opt.gamma),
            rm.stretch(ref, lim_ref, opt.gamma),
            opt.palette,
        )
        if opt.numbers == "none":
            note, vals = "", {}
        elif qc is None:
            note, vals = rm.image_note(ref, mov, px)
        elif name == "before":
            vals = {
                "source": "scorer",
                "stage": rm.NATIVE_STAGE,
                "dice_matched": qc.native_dice,
            }
            note = f"Dice = {qc.native_dice:.2f}" if qc.native_dice is not None else ""
        else:
            note, vals = rm.cell_note(qc, y, x, field_px, px, 5)
            vals = {"source": "scorer", **vals}
        panels[name] = (img, note, vals)

    scalebar = None
    if px and opt.scalebar_um != 0:
        bar_um = opt.scalebar_um or rm.auto_scalebar_um(field_px * px)
        scalebar = (bar_um / px, f"{bar_um:g} µm")
    formats = [s.strip() for s in opt.formats.split(",") if s.strip()]
    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{arm.patient}_{key}"
    legend = rm.PALETTE_LEGEND.get(opt.palette, "")
    for name, title in (
        ("before", "Before"),
        ("after", f"After ({opt.title or arm.name})"),
    ):
        img, note, _ = panels[name]
        draw_panel(
            img,
            title,
            note,
            scalebar,
            outdir / f"{stem}_{name}",
            formats,
            opt.dpi,
            legend,
        )
    low, f = comp.lowres_reference(max(1, int(round(opt.lowres_um / px))) if px else 16)
    rm.save_locator(
        low,
        f,
        [(y, x)],
        field_px,
        outdir / f"{stem}_locator",
        formats,
        px,
        f"{arm.patient} — {arm.ref.label} (reference)",
        opt.dpi,
    )
    manifest = {
        "patient": arm.patient,
        "round": key,
        "moving": str(sl.image),
        "reference": str(arm.ref.image),
        "composite": str(comp.src.path),
        "pixel_size_um": px,
        "crop": {
            "y": y,
            "x": x,
            "size_px": field_px,
            "size_um": field_px * px if px else None,
        },
        "avoided_rois_from": str(opt.avoid_rois_json) if opt.avoid_rois_json else None,
        "palette": opt.palette,
        "pixels": {"before": used_b, "after": used_a},
        "stretch": {"pmin": opt.pmin, "pmax": opt.pmax, "gamma": opt.gamma},
        "scalebar_um": scalebar and float(scalebar[1].split()[0]),
        "numbers": {k: v[2] for k, v in panels.items()},
    }
    (outdir / f"{stem}_overlay.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "arm",
        type=Path,
        metavar="RUN_DIR",
        help="a mirage --outdir that ran registration",
    )
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument(
        "--patient",
        default=None,
        help="default: the only patient in csv/registered.csv",
    )
    ap.add_argument(
        "--rounds", nargs="*", default=None, help="moving round(s); default: every one"
    )
    ap.add_argument(
        "--title",
        default=None,
        help="method name in the After title (default: the dir name)",
    )
    ap.add_argument("--field-um", type=float, default=500.0, help="crop side in µm")
    ap.add_argument(
        "--field-px",
        type=int,
        default=None,
        help="crop side in px (overrides --field-um)",
    )
    ap.add_argument(
        "--roi",
        default=None,
        metavar="Y,X",
        help="crop top-left, reference frame, full-res px",
    )
    ap.add_argument(
        "--avoid-rois-json",
        type=Path,
        default=None,
        help="a mosaic's *_rois.json to stay clear of",
    )
    ap.add_argument("--palette", choices=list(rm.PALETTES), default="magenta-cyan")
    ap.add_argument(
        "--numbers", choices=("auto", "scorer", "image", "none"), default="auto"
    )
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.8)
    ap.add_argument("--gamma", type=float, default=1.0, help="<1 lifts the background")
    ap.add_argument(
        "--source",
        choices=("auto", "originals", "composite"),
        default="auto",
        help="pixels from the original 16-bit slides or the 8-bit QC composite (see reg_mosaic)",
    )
    ap.add_argument("--native-csv", type=Path, default=None)
    ap.add_argument(
        "--scalebar-um",
        type=float,
        default=None,
        help="default auto ≈ field/4; 0 = none",
    )
    ap.add_argument("--pixel-size-um", type=float, default=None)
    ap.add_argument("--lowres-um", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None) -> int:
    opt = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if opt.verbose else logging.INFO, format="%(message)s"
    )
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    arm = rm.Arm(opt.arm, source=opt.source, native_csv=opt.native_csv)
    patients = arm.patients()
    pid = opt.patient or (patients[0] if len(patients) == 1 else None)
    if pid is None:
        raise SystemExit(
            f"{len(patients)} patients in {opt.arm}; pass --patient ({patients})"
        )
    arm.open(pid)
    keys = [
        k
        for k, sl in arm.moving.items()
        if not opt.rounds or rm.round_matches(sl, opt.rounds)
    ]
    if not keys:
        raise SystemExit(f"{pid}: no moving round matches {opt.rounds}")
    for key in keys:
        render(arm, key, opt)
    arm.close()
    log.info("wrote %d before/after pair(s) in %s", len(keys), opt.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
