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
show different tissue. ``--variants N`` draws N pairs per round instead of one --
<patient>_<round>_v1_before, _v2_... -- each on tissue no earlier variant used, so a crop
that happens to read badly is not the only output. (N = 1, the default, keeps the names
above.)

THE ZOOM. ``--zoom-um N`` draws each panel beside a framed zoom of an N µm region, READ AT
FULL RESOLUTION, so individual cells stay visible in a field too wide to resolve them (a
2 mm panel is strided down to --max-px; the zoom is not). The layout is reg_zoom's, from
the one shared implementation (reg_mosaic.draw_overview_zoom): crop left, zoom top right,
white box and funnel between, channel names bottom right, a scale bar on each. The region
is the most textured window inside the crop unless --zoom-roi names one.

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
    img,
    title,
    note,
    scalebar,
    out_stem: Path,
    formats,
    dpi,
    legend,
    size_in=None,
    zoom=None,
):
    """One panel: the crop, or -- with ``zoom`` -- the crop beside a framed zoom of it.

    The zoom layout is reg_mosaic.draw_overview_zoom, the same function reg_zoom draws its
    segmentation figure with, so the two look like one family: image left, framed zoom top
    right, funnel between, channel names bottom right under the zoom.
    """
    if zoom is not None:
        return rm.draw_overview_zoom(
            img,
            zoom["img"],
            zoom["box"],
            zoom["factor"],
            zoom["px"],
            out_stem,
            formats,
            dpi=dpi,
            zoom_fraction=zoom["fraction"],
            funnel_alpha=zoom["funnel_alpha"],
            legend=legend,
            title=title,
            note=note,
            frame_color=zoom["color"],
            bar_over_um=scalebar[2] if scalebar else None,
        )
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
                0.97,
                note,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=font * 0.85,
                color="white",
            )
        )
    if scalebar:
        rm.draw_scalebar(ax, h, w, scalebar[0], scalebar[1], font * 0.85, thick=0.01)
    if (
        legend
    ):  # channel names in their colours, lower right (the numbers sit top right)
        rm.draw_legend(ax, legend, font * 1.1)
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


def zoom_region(arm: rm.Arm, key: str, opt, crop, ref_panel, step: int, px):
    """The inset's region and its FULL-resolution pixels, or None when --zoom-um is off.

    The region is chosen inside the crop by the same tissue x texture score that picked the
    crop itself (on the panel already in memory, so no extra read), unless --zoom-roi names
    it. Both panels are read at step 1: the point of the inset is to show cells the strided
    panel cannot resolve.
    """
    if not opt.zoom_um:
        return None
    y, x, field_px = crop
    if not px:
        raise SystemExit("--zoom-um needs a pixel size; pass --pixel-size-um")
    size = int(round(opt.zoom_um / px))
    if size < 8:
        raise SystemExit(f"--zoom-um {opt.zoom_um} is {size} px: too small to see")
    if size >= field_px:
        log.warning(
            "--zoom-um %g is %d px, the whole %d px crop: clamping to a quarter of it",
            opt.zoom_um,
            size,
            field_px,
        )
        size = max(8, field_px // 4)
    if opt.zoom_roi:
        zy, zx = (int(v) for v in opt.zoom_roi.split(",")[:2])
    else:
        picks = rm.select_rois(ref_panel, step, size, 1, 0.0, 0.0, (field_px, field_px))
        dy, dx = picks[0] if picks else ((field_px - size) // 2,) * 2
        zy, zx = y + dy, x + dx
    zy = min(max(zy, y), y + field_px - size)
    zx = min(max(zx, x), x + field_px - size)
    crops = {}
    for panel in ("after", "before"):
        zr, zm, _ = arm.crop(key, panel, zy, zx, size, size, 1)
        crops[panel] = (zr, zm)
    bar = None
    if opt.scalebar_um != 0:
        bar_um = rm.auto_scalebar_um(size * px)
        bar = (bar_um / px, rm.scalebar_label(bar_um), bar_um)
    log.info(
        "%s %s: zoom y=%d x=%d size=%d px = %g µm (full resolution)",
        arm.patient,
        key,
        zy,
        zx,
        size,
        size * px,
    )
    return {
        "y": zy,
        "x": zx,
        "size_px": size,
        "size_um": size * px,
        "crops": crops,
        "scalebar": bar,
        "panel": {},
    }


def render(
    arm: rm.Arm, key: str, opt: argparse.Namespace, exclude=(), tag: str = ""
) -> dict:
    """One Before/After pair for one round.

    ``exclude`` is [(y, x, size_px), ...] an earlier variant already drew (on top of
    --avoid-rois-json), ``tag`` is appended to every output name (``_v2``).
    """
    sl = arm.slide(key)
    # The canvas, the crop choice and the locator come from the reference: from the QC
    # composite when the run wrote one, else from the original reference slide itself.
    orig = None if opt.source == "composite" else arm.originals(key)
    comp = None
    if arm.composite_path(key) is not None or orig is None:
        comp = arm.composite(key)
        if not comp.has_before:
            raise SystemExit(
                f"{comp.src.path.name}: no Before panel (QC ran without --native)"
            )
    ref_src, ref_ci = (orig[0], orig[1]) if orig else (None, None)
    H, W = comp.canvas if comp else ref_src.shape
    px = opt.pixel_size_um or (comp.px if comp else ref_src.px) or arm.px

    def lowres(hint_um):
        if comp is not None:
            return comp.lowres_reference(max(1, int(round(hint_um / px))) if px else 16)
        return ref_src.overview(ref_ci, 2400)

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
        low, f = lowres(opt.lowres_um)
        picks = rm.select_rois(
            low,
            f,
            field_px,
            1,
            0.0,
            0.0,
            (H, W),
            exclude=[*load_avoid(opt.avoid_rois_json, arm.ref.image), *exclude],
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

    # a big field (e.g. 10 mm = ~31k px) is read strided to at most --max-px per side
    step = max(1, -(-field_px // opt.max_px))
    ref_a, mov_a, used_a = arm.crop(key, "after", y, x, field_px, field_px, step)
    ref_b, mov_b, used_b = arm.crop(key, "before", y, x, field_px, field_px, step)

    # --zoom-um: a small inset read at FULL resolution (step 1), so individual nuclei are
    # visible even when the panel itself is strided down from a millimetre-scale field
    zoom = zoom_region(arm, key, opt, (y, x, field_px), ref_a, step, px)
    # one stretch for the reference (it is the same pixels in both panels); the moving
    # channel is stretched per panel, since the two crops cover different tissue
    lim_ref = rm.percentile_limits(ref_a, opt.pmin, opt.pmax)
    # the inset is a different (full-resolution) read, so it gets its own reference limits
    lim_ref_zoom = (
        rm.percentile_limits(zoom["crops"]["after"][0], opt.pmin, opt.pmax)
        if zoom is not None
        else None
    )
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
        if zoom is not None:
            zr, zm = zoom["crops"][name]
            # box and factor in the drawn panel's frame: the panel is strided by `step`,
            # so draw_overview_zoom divides the full-res box by it exactly as it does for
            # reg_zoom's downsampled overview
            zoom["panel"][name] = {
                "img": rm.overlay(
                    rm.stretch(
                        zm, rm.percentile_limits(zm, opt.pmin, opt.pmax), opt.gamma
                    ),
                    rm.stretch(zr, lim_ref_zoom, opt.gamma),
                    opt.palette,
                ),
                "box": (zoom["y"] - y, zoom["x"] - x, zoom["size_px"]),
                "factor": step,
                "px": px,
                "fraction": opt.zoom_frac,
                "funnel_alpha": opt.funnel_alpha,
                "color": opt.zoom_color,
            }
        if opt.numbers == "none":
            note, vals = "", {}
        elif qc is None:
            note, vals = rm.image_note(ref, mov, px * step if px else px)
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
        scalebar = (bar_um / (px * step), rm.scalebar_label(bar_um), bar_um)
    formats = [s.strip() for s in opt.formats.split(",") if s.strip()]
    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{arm.patient}_{key}{tag}"
    moving_rgb, reference_rgb = rm.PALETTES[opt.palette]
    legend = [("reference DAPI", reference_rgb), ("moving DAPI", moving_rgb)]
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
            zoom=zoom["panel"][name] if zoom is not None else None,
        )
    low, f = lowres(opt.lowres_um)
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
        "variant": int(tag[2:]) if tag else 1,
        "moving": str(sl.image),
        "reference": str(arm.ref.image),
        "composite": str(comp.src.path) if comp else None,
        "pixel_size_um": px,
        "crop": {
            "step": step,
            "y": y,
            "x": x,
            "size_px": field_px,
            "size_um": field_px * px if px else None,
        },
        "zoom": None
        if zoom is None
        else {
            "y": zoom["y"],
            "x": zoom["x"],
            "size_px": zoom["size_px"],
            "size_um": zoom["size_um"],
            "step": 1,
            "fraction": opt.zoom_frac,
            "color": opt.zoom_color,
        },
        "avoided_rois_from": str(opt.avoid_rois_json) if opt.avoid_rois_json else None,
        "palette": opt.palette,
        "pixels": {"before": used_b, "after": used_a},
        "stretch": {"pmin": opt.pmin, "pmax": opt.pmax, "gamma": opt.gamma},
        "scalebar_um": scalebar and float(scalebar[2]),
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
    ap.add_argument(
        "--variants",
        type=int,
        default=1,
        help="draw N Before/After pairs per round on different tissue, "
        "<pid>_<round>_v1_before, _v2_... (default 1: the current names). Each variant "
        "avoids every earlier one's crop, so they are alternatives to choose between. "
        "Ignored with --roi, which fixes the crop.",
    )
    ap.add_argument(
        "--zoom-um",
        type=float,
        default=0.0,
        help="draw each panel beside a framed zoom of this many µm, READ AT FULL "
        "RESOLUTION, in reg_zoom's layout: crop left, zoom top right, funnel between, "
        "channel names bottom right -- to see individual cells in a field too wide to "
        "resolve them (0 = no zoom, the default). E.g. --field-um 2000 --zoom-um 60.",
    )
    ap.add_argument(
        "--zoom-roi",
        default=None,
        help='"Y,X" top-left of the inset in the REFERENCE frame (full-res px); '
        "default: the most textured window inside the crop",
    )
    ap.add_argument(
        "--zoom-frac",
        type=float,
        default=0.62,
        help="zoom panel height as a fraction of the crop's (default 0.62, as reg_zoom)",
    )
    ap.add_argument(
        "--funnel-alpha",
        type=float,
        default=0.18,
        help="opacity of the funnel joining the box to the zoom (default 0.18)",
    )
    ap.add_argument(
        "--zoom-color",
        default="white",
        help="colour of the zoom frame and of its box on the crop (default white)",
    )
    ap.add_argument("--field-um", type=float, default=500.0, help="crop side in µm")
    ap.add_argument(
        "--field-px",
        type=int,
        default=None,
        help="crop side in px (overrides --field-um)",
    )
    ap.add_argument(
        "--max-px",
        type=int,
        default=4096,
        help="largest image side drawn; a bigger field is read with a stride (a 10 mm field "
        "at 0.325 um/px is ~31k px), so memory stays bounded and the scale bar stays exact",
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
    n = rm.variant_count(opt)
    drawn = 0
    for key in keys:
        # variant 2 avoids variant 1's crop, and so on: N alternatives, not N copies
        exclude: list[tuple[int, int, int]] = []
        for v in range(1, n + 1):
            try:
                m = render(
                    arm, key, opt, exclude=exclude, tag="" if n == 1 else f"_v{v}"
                )
            except SystemExit as exc:
                # the slide ran out of tissue no earlier variant used: keep what was drawn
                if v == 1:
                    raise
                log.warning(
                    "%s %s: stopping at %d variant(s) (%s)", pid, key, v - 1, exc
                )
                drawn += v - 1
                break
            c = m["crop"]
            exclude.append((c["y"], c["x"], c["size_px"]))
        else:
            drawn += n
    arm.close()
    log.info("wrote %d before/after pair(s) in %s", drawn, opt.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
