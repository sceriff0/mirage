"""HyReCo, the external landmark set: re-stained H&E / PHH3 pairs as benchmark cases.

HyReCo (van der Laak, Lotz, Weiss, Heldmann, IEEE DataPort 2021, doi:10.21227/pzj5-bs61;
reference publication Lotz et al. 2023, J Med Imaging 10(6):067501,
doi:10.1117/1.JMI.10.6.067501) is the one public set found where the SAME section was
imaged twice (H&E, bleached, re-stained PHH3, scanned again) and annotated by hand.
Treated as CC BY-SA 4.0 (page title and paper; the DataCite record says CC BY 4.0).

**One dataset** (ruling 2026-10-19). ``prepare hyreco --data-root DIR`` writes ``hyreco``:
subset A's nine re-stained pairs (``DIR/HyReCo/{HE,PHH3}``) and, when it is there, subset
B's 54 ("HyReCo-Additional", ``DIR/HyReCo-Additional/{HE,PHH3}``; ``download/hyreco.sh``
fetches it by default, and ``HYRECO_ADDITIONAL=0`` fetches the core zip alone), in one
cases folder, scored together.
A case id is the file's id prefixed by its subset (``a_29``, ``b_29``): four ids (29,
108, 361, 464) are in both lists, whether they are the same slides is not stated by any
source read, so both are kept, distinct, and flagged (``extra.shared_id``; a column of
the pair table and a count in the summary). Nothing is dropped for it.

Nothing is tuned (no lock): every run of the list runs with its own fixed settings, and
the proxy settings below (``--proxy``, ``--reduce``, default luminance at native
resolution) are pre-declared and recorded on every case (``settings`` /
``settings_sha256``); the scorer names them when two cases differ.

A subset folder is accepted only when its file ids are that subset's (the lists below,
from the dataset page), so one subset's files cannot be prepared under the other's name.

**Roles.** H&E is the reference (scanned first, like cycle 1), PHH3 the moving slide; each
pair is its own patient (``group = case_id``). The landmarks of the two slides, in
full-resolution pixels and paired by line, are the case's ``truth.npz``
(``regbench.landmark_io``: ``ref`` the H&E slide's, ``mov`` the PHH3 slide's), written
before ``case.json``.

**What a method reads.** ``ref_image`` / ``mov_image`` are the ORIGINAL RGB TIFFs.
``ref_nuclear`` / ``mov_nuclear`` are a one-channel proxy of each, written here, for a
method that takes one channel (STARE): benchmark-side preprocessing, recorded as such.

``--proxy luminance``     255 - (0.299 R + 0.587 G + 0.114 B), uint8. The pre-declared
                          default: the dataset's authors feed their own method "color to
                          gray scale and inverted to obtain a black background" (Lotz et
                          al. 2023, section 2.2).
``--proxy haematoxylin``  the first channel of ``skimage.color.rgb2hed`` (Ruifrok and
                          Johnston's matrix), clipped to [0, 6.5535] and stored as uint16
                          at 10000 counts per unit. It assumes the PHH3 slide has a
                          haematoxylin counterstain, which no source read states. Not
                          the default and not used by the run list.
``--reduce 2``            the proxy block-averaged 2 x 2 (0.48 um per pixel); 1 = native.
                          Not the default. Landmarks and ``ref_hw`` stay in
                          full-resolution pixels; a method given the proxy scales its map
                          (``extra["settings"]["reduce"]``, ``methods/stare.py``).

**The layout, and what is verified.** From the dataset page (fetched 2026-10-10, whole
page text read, no data file opened: ``[FULL: page; the archives were never opened]``)::

    HyReCo/{CD45,CD8,HE,KI67,PHH3}/<id>.tif|.csv       subset A, HyReCo.zip (233 GB)
    HyReCo-Additional/HE/<id>.tif|.csv                 subset B, HyReCo-Additional-HE.zip
    HyReCo-Additional/PHH3/<id>.tif|.csv               subset B, HyReCo-Additional-PHH3.zip

Verified there: the two id lists; BigTIFF plus one CSV per image; "Each line describes a
point annotation in world coordinates (x,y,z). The unit is mm. The origin of the
coordinate system is the upper left corner of the image."; the lists "attempt to mark the
same location in all images of a single case". From the paper (Europe PMC text of
PMC10704256, ``[PARTIAL: the lines matching a search for the pixel size, the landmarks,
the grey-scale conversion, MTRE and the observer errors; figures and supplement not
seen]``): "0.24 um/px and ~95,000 x 220,000 pixels".

**Assumed, because no file was opened; each is checked on the files.** Nothing blocks
(ruling 2026-10-18): a pair whose files are missing or unreadable is not prepared and is a
row of ``prepare_dropped.csv``; a pair that fails a check below is prepared, scored and
FLAGGED (``extra.flags``; ``losses.csv``: ``landmark_flagged``), never removed.

1. *The first TIFF series' first level is the full-resolution image, RGB, uint8.*
   Anything else drops the pair (``_open_rgb``): there is nothing to register.
2. *Landmarks are paired by line number.* The page implies it and does not say it. Two
   CSVs of a pair with different numbers of points are FLAGGED and the common prefix is
   scored (the rule BIRL's evaluation applies). A line that is not ``x,y[,z]``
   drops the pair wherever it is, except one header line at the top: a blank line
   between points could be a missing landmark, and an id column (more than three
   columns) must be paired by id, which this module does not do.
3. *Pixel = millimetres / pixel size*, with no half-pixel term. Whether "upper left
   corner" is the corner of the first pixel or its centre is not stated; the same
   convention on both slides moves both landmarks of a pair by the same half pixel.
4. *The pixel size.* The conversion needs more than the paper's two digits: a landmark
   50 mm from the origin moves by 2000 px per 1 % of pixel size. So the FILE's value is
   used when it has one (resolution tags in inches or centimetres, or an ``MPP = `` field
   in the image description); one more than 2 % off the paper's 0.24 is FLAGGED. A file
   with no value falls back to 0.24 exactly and says so (``pixel_size_source``; the
   summary repeats it, because two digits are then all the millimetre conversion rests
   on). ``--pixel-size-um`` overrides both, under the same check. Two slides of a pair
   more than 0.1 % apart are FLAGGED.
5. *Every landmark lies inside its image* after conversion; one outside is FLAGGED: the
   pixel size or the origin may be wrong.

Never run on the real files. Look at the first prepared case before any array is
submitted: ``python -m benchmarks.regbench.datasets.hyreco --cases C --dataset hyreco
--case-id a_29 --out DIR`` writes, for the first landmarks of the pair, the crop of each
slide around its landmark, side by side (H&E left, PHH3 right) with a cross on the
landmark. The two crosses must sit on the same nucleus.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .. import landmark_io
from ..cases import Case, case_dir, write_case
from ..imageio import _rgb_view, write_nuclear_tiles

# The dataset page's two file listings (module docstring). Ids are file stems.
SUBSET_A_IDS = ("29", "108", "361", "464", "533", "611", "628", "644", "679")
SUBSET_B_IDS = (
    "8", "29", "31", "43", "61", "64", "67", "79", "82", "108", "128", "147", "155",
    "157", "158", "168", "172", "208", "237", "241", "244", "245", "260", "264", "272",
    "281", "296", "313", "315", "320", "343", "349", "359", "361", "364", "385", "397",
    "399", "400", "405", "409", "446", "449", "457", "464", "485", "500", "504", "506",
    "514", "529", "541", "663", "705",
)  # fmt: skip
SUBSET_IDS = {"a": SUBSET_A_IDS, "b": SUBSET_B_IDS}
SHARED_IDS = tuple(i for i in SUBSET_B_IDS if i in SUBSET_A_IDS)
HYRECO = "hyreco"
# subset -> its folder under --data-root (the archives' own top folders)
SUBSET_FOLDER = {"a": "HyReCo", "b": "HyReCo-Additional"}
REF_STAIN, MOV_STAIN = "HE", "PHH3"

PIXEL_SIZE_UM = 0.24  # Lotz et al. 2023, two digits
PIXEL_SIZE_TOLERANCE = 0.02  # a file's or a given value against the paper's
PAIR_PIXEL_TOLERANCE = 0.001  # the two slides of a pair against each other
PROXIES = ("luminance", "haematoxylin")
DEFAULT_PROXY = "luminance"
REDUCTIONS = (1, 2)
LUMA = (0.299, 0.587, 0.114)
HAEMATOXYLIN_MAX, HAEMATOXYLIN_SCALE = 6.5535, 10000.0
TILE = 1024
SETTINGS_VERSION = 3  # 3: the tuning lock and keep_shared_ids are gone (2026-10-18)

LICENCE = "CC BY-SA 4.0 (dataset page and paper; DataCite says CC BY 4.0)"
CITATION = (
    "van der Laak, Lotz, Weiss, Heldmann, HyReCo, IEEE DataPort 2021, "
    "doi:10.21227/pzj5-bs61; Lotz et al. 2023, doi:10.1117/1.JMI.10.6.067501"
)


class Refused(ValueError):
    """A case this module will not prepare, with the reason."""


# ── settings ──────────────────────────────────────────────────────────────────
def make_settings(proxy=None, reduce=None, pixel_size_um=None):
    """Every choice that reaches a result: the proxy, its reduction and a stated pixel
    size (None: the file's own)."""
    proxy = DEFAULT_PROXY if proxy is None else str(proxy)
    reduce = 1 if reduce is None else int(reduce)
    if proxy not in PROXIES:
        raise SystemExit(f"--proxy {proxy!r}: expected one of {PROXIES}")
    if reduce not in REDUCTIONS:
        raise SystemExit(f"--reduce {reduce!r}: expected one of {REDUCTIONS}")
    if pixel_size_um is not None:
        pixel_size_um = float(pixel_size_um)
        if not pixel_size_um > 0:
            raise SystemExit(
                f"--pixel-size-um {pixel_size_um!r}: expected a positive number"
            )
    return {
        "proxy": proxy,
        "reduce": reduce,
        "version": SETTINGS_VERSION,
        "pixel_size_um": pixel_size_um,
    }


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


# ── landmarks ─────────────────────────────────────────────────────────────────
def _fields(line):
    return [t for t in re.split(r"[,;\s]+", line.strip()) if t]


def _numbers(tokens):
    try:
        values = [float(t) for t in tokens]
    except ValueError:
        return None
    return values if all(math.isfinite(v) for v in values) else None


def read_points_mm(path):
    """``((N, 2) x, y in millimetres, has_header)`` of one landmark file.

    One point per line, ``x,y`` or ``x,y,z``, separated by commas, semicolons or blanks.
    The first line may be a header. Anything else is refused with the file and the line:
    points are paired with the other slide's by position, so a skipped line would pair
    every later landmark with the wrong partner.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise Refused(f"{path}: landmark file not readable ({exc})") from None
    while lines and not lines[-1].strip():
        lines.pop()
    out, header = [], False
    for n, line in enumerate(lines, 1):
        tokens = _fields(line)
        values = _numbers(tokens) if tokens else None
        if values is None:
            if n == 1 and tokens:
                header = True
                continue
            what = "a blank line" if not tokens else "not numbers"
            raise Refused(f"{path}:{n}: {what} ({line!r}); expected x,y[,z] in mm")
        if len(values) not in (2, 3):
            raise Refused(
                f"{path}:{n}: {len(values)} columns ({line!r}); expected x,y[,z]. A "
                "file with an id column must be paired by id, which this module does "
                "not do"
            )
        out.append(values[:2])
    if not out:
        raise Refused(f"{path}: no landmarks")
    return np.array(out, dtype=float).reshape(-1, 2), header


# ── the slides ────────────────────────────────────────────────────────────────
def _open_rgb(tif, path):
    try:
        level = tif.series[0].levels[0]
        read, hw = _rgb_view(level)
    except (ValueError, IndexError) as exc:
        raise Refused(f"{path}: not an RGB image in its first series ({exc})") from None
    if np.dtype(level.dtype) != np.uint8:
        raise Refused(f"{path}: expected 8-bit RGB, found {level.dtype}")
    return read, hw


def file_pixel_size_um(path):
    """The pixel size a TIFF states for its full-resolution image, in micrometres, or
    None: its resolution tags (inches or centimetres), else ``MPP = `` in the image
    description. A file whose X and Y values differ is refused."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        page = tif.series[0].levels[0].pages[0]
        tags = page.tags
        unit = tags.get("ResolutionUnit")
        per_unit = {2: 25400.0, 3: 10000.0}.get(int(unit.value)) if unit else None
        found = []
        for key in ("XResolution", "YResolution"):
            tag = tags.get(key)
            if tag is None or per_unit is None:
                continue
            num, den = tag.value
            if num and den:
                found.append(per_unit * den / num)
        if len(found) == 2:
            if abs(found[0] - found[1]) > 1e-3 * found[0]:
                raise Refused(
                    f"{path}: pixels are {found[0]:.5f} x {found[1]:.5f} um; one pixel "
                    "size is assumed"
                )
            return float(found[0])
        m = re.search(r"\bMPP\s*=\s*([0-9.]+)", str(page.description or ""))
        return float(m.group(1)) if m else None


def pixel_size(path, given=None):
    """``(um per pixel, source, what the file states or None, flag or "")`` of one slide
    (module docstring, assumption 4)."""
    stated = file_pixel_size_um(path)
    if given:
        value, source = float(given), "command line (--pixel-size-um)"
    elif stated:
        value, source = stated, "file"
    else:
        value = PIXEL_SIZE_UM
        source = "paper, two digits (the file states none)"
    flag = ""
    if abs(value - PIXEL_SIZE_UM) > PIXEL_SIZE_TOLERANCE * PIXEL_SIZE_UM:
        flag = (
            f"{Path(path).name}: pixel size {value:.5f} um ({source}) is not within "
            f"{PIXEL_SIZE_TOLERANCE:.0%} of the published {PIXEL_SIZE_UM} um"
        )
    return value, source, stated, flag


def proxy_plane(band, proxy):
    """One band of RGB (float32, 0..255) as the one-channel proxy: nuclei bright."""
    if proxy == "luminance":
        lum = band @ np.asarray(LUMA, np.float32)
        return (255.0 - lum).round().clip(0, 255).astype(np.uint8)
    from skimage.color import rgb2hed

    hed = rgb2hed(np.clip(band, 0, 255) / 255.0)[..., 0]
    return (
        (np.clip(hed, 0.0, HAEMATOXYLIN_MAX) * HAEMATOXYLIN_SCALE)
        .round()
        .astype(np.uint16)
    )


PROXY_DTYPE = {"luminance": np.uint8, "haematoxylin": np.uint16}


def write_proxy(tif_path, out, settings, pixel_size_um, force=False):
    """The proxy of one slide as a tiled, compressed, one-channel OME-TIFF.
    Returns ``((H, W) of the slide, (h, w) of the proxy)``."""
    import tifffile

    proxy, f = settings["proxy"], int(settings["reduce"])
    with tifffile.TiffFile(str(tif_path)) as tif:
        read, (h, w) = _open_rgb(tif, tif_path)
        ph, pw = h // f, w // f
        if min(ph, pw) < 1:
            raise Refused(f"{tif_path}: {h} x {w} px is too small to reduce by {f}")
        if Path(out).exists() and not force:
            return (h, w), (ph, pw)

        def tiles():
            for y0 in range(0, ph, TILE):
                y1 = min(y0 + TILE, ph)
                band = read(y0 * f, y1 * f)[:, : pw * f].astype(np.float32)
                if f > 1:
                    band = band.reshape(y1 - y0, f, pw, f, 3).mean(axis=(1, 3))
                plane = proxy_plane(band, proxy)
                for x0 in range(0, pw, TILE):
                    yield plane[:, x0 : x0 + TILE]

        write_nuclear_tiles(
            out, (ph, pw), PROXY_DTYPE[proxy], tiles(), TILE, pixel_size_um * f
        )
    return (h, w), (ph, pw)


def _landmarks_px(tif, hw, px_um):
    csv = Path(tif).with_suffix(".csv")
    mm, header = read_points_mm(csv)
    xy = mm * 1000.0 / px_um
    h, w = hw
    outside = int(
        np.sum(
            (xy[:, 0] < -0.5)
            | (xy[:, 1] < -0.5)
            | (xy[:, 0] > w - 0.5)
            | (xy[:, 1] > h - 0.5)
        )
    )
    flag = ""
    if outside:
        flag = (
            f"{csv.name}: {outside} landmark(s) outside the {w} x {h} px image at "
            f"{px_um:.5f} um/px (x {xy[:, 0].min():.0f}..{xy[:, 0].max():.0f}, "
            f"y {xy[:, 1].min():.0f}..{xy[:, 1].max():.0f}): the pixel size or the "
            "origin may not be what this module assumes"
        )
    return xy, header, csv, flag


def prepare_one(
    cases_root,
    subset,
    cid,
    ref_dir,
    mov_dir,
    settings,
    pixel_size_um=None,
    force=False,
    complete=True,
):
    dataset = HYRECO
    case_id = f"{subset}_{cid}"
    d = case_dir(cases_root, dataset, case_id)
    if (d / "case.json").exists() and not force:
        return d
    ref_t, mov_t = Path(ref_dir) / f"{cid}.tif", Path(mov_dir) / f"{cid}.tif"
    for p in (ref_t, mov_t, ref_t.with_suffix(".csv"), mov_t.with_suffix(".csv")):
        if not p.is_file():
            raise Refused(f"{p}: missing")
    f = int(settings["reduce"])
    tag = f"{settings['proxy']}.r{f}"
    cache = Path(cases_root) / "_cache" / dataset
    ref_n = cache / subset / REF_STAIN / f"{cid}.{tag}.ome.tif"
    mov_n = cache / subset / MOV_STAIN / f"{cid}.{tag}.ome.tif"
    # landmarks before the proxies: a dropped pair costs no whole-slide read
    import tifffile

    hws = []
    for t in (ref_t, mov_t):
        with tifffile.TiffFile(str(t)) as tif:
            hws.append(_open_rgb(tif, t)[1])
    ref_px, ref_source, ref_stated, ref_flag = pixel_size(ref_t, pixel_size_um)
    mov_px, mov_source, mov_stated, mov_flag = pixel_size(mov_t, pixel_size_um)
    flags = [x for x in (ref_flag, mov_flag) if x]
    if abs(ref_px - mov_px) > PAIR_PIXEL_TOLERANCE * ref_px:
        flags.append(
            f"the reference is {ref_px:.5f} um/px and the moving slide {mov_px:.5f}: "
            "one pixel size per pair is assumed"
        )
    ref_all, ref_header, ref_csv, ref_out = _landmarks_px(ref_t, hws[0], ref_px)
    mov_all, mov_header, mov_csv, mov_out = _landmarks_px(mov_t, hws[1], mov_px)
    flags += [x for x in (ref_out, mov_out) if x]
    n = min(len(ref_all), len(mov_all))
    if len(ref_all) != len(mov_all):
        flags.append(
            f"{len(ref_all)} landmarks in {ref_csv.name} ({REF_STAIN}) and "
            f"{len(mov_all)} in {mov_csv.name} ({MOV_STAIN}): paired by line, the "
            "common prefix is scored"
        )
    ref_xy, mov_xy = ref_all[:n], mov_all[:n]
    _, ref_proxy_hw = write_proxy(ref_t, ref_n, settings, ref_px, force)
    _, mov_proxy_hw = write_proxy(mov_t, mov_n, settings, mov_px, force)
    case = Case(
        dataset=dataset,
        case_id=case_id,
        group=case_id,  # each pair is its own patient
        ref_nuclear=str(ref_n),
        mov_nuclear=str(mov_n),
        ref_image=str(ref_t),
        mov_image=str(mov_t),
        modality="brightfield",
        diagonal=math.hypot(*hws[0]),
        pixel_size_um=ref_px,
        ref_hw=list(hws[0]),
        mov_hw=list(hws[1]),
        extra={
            "subset": subset.upper(),
            "file_id": str(cid),
            # an id both subsets' lists have: kept in both, flagged (ruling 2026-10-19)
            "shared_id": str(cid) in SHARED_IDS,
            "subset_complete": bool(complete),
            "stain_ref": REF_STAIN,
            "stain_mov": MOV_STAIN,
            "n_landmarks": int(n),
            "n_landmarks_file": {"ref": int(len(ref_all)), "mov": int(len(mov_all))},
            "landmark_files": {"ref": str(ref_csv), "mov": str(mov_csv)},
            "landmark_header": {"ref": ref_header, "mov": mov_header},
            "landmark_pairing": "by line (assumed; the page implies it); the common "
            "prefix where the two files differ in length",
            "landmark_origin": "px = mm / pixel size, no half-pixel term (assumed)",
            "pixel_size_source": ref_source,
            "pixel_size_file_um": {"ref": ref_stated, "mov": mov_stated},
            "pixel_size_mov_um": mov_px,
            "flags": flags,
            "settings": dict(settings),
            "settings_sha256": settings_hash(settings),
            "proxy_hw": {"ref": list(ref_proxy_hw), "mov": list(mov_proxy_hw)},
            "truth": "manual landmarks (Fraunhofer MEVIS)",
            "licence": LICENCE,
            "citation": CITATION,
        },
    )
    # the landmarks first: case.json, written last, says the case is complete
    landmark_io.write_truth(d, ref_xy, mov_xy)
    return write_case(cases_root, case)


# ── the subset ────────────────────────────────────────────────────────────────
def _stems(folder):
    return {p.stem for p in Path(folder).glob("*.tif")}


def subset_ids(subset, ref_dir, mov_dir):
    """``(ids to prepare in the page's list order, missing ids)`` after checking that the
    two folders ARE this subset: stain folder names, no id of another subset."""
    ref_dir, mov_dir = Path(ref_dir), Path(mov_dir)
    for folder, stain in ((ref_dir, REF_STAIN), (mov_dir, MOV_STAIN)):
        if not folder.is_dir():
            raise SystemExit(f"{folder}: no such folder")
        if folder.name.upper() != stain:
            raise SystemExit(
                f"{folder}: expected the dataset's `{stain}` folder (H&E is the "
                "reference, PHH3 the moving slide)"
            )
    wanted = SUBSET_IDS[subset]
    found = _stems(ref_dir) | _stems(mov_dir)
    foreign = sorted(found - set(wanted), key=lambda s: (len(s), s))
    if foreign:
        raise SystemExit(
            f"{ref_dir.parent}: this is not subset {subset.upper()}: it holds "
            f"{len(foreign)} case(s) that subset does not have "
            f"({', '.join(foreign[:8])}{', ...' if len(foreign) > 8 else ''}). One "
            "subset's files are never prepared under the other's name."
        )
    have = _stems(ref_dir) & _stems(mov_dir)
    return [i for i in wanted if i in have], [i for i in wanted if i not in have]


def subsets_there(data_root):
    """``[(subset, HE folder, PHH3 folder)]`` of the subsets under ``data_root``:
    subset A always (refused when absent), subset B when its folder is there."""
    root = Path(data_root)
    out = []
    for subset, folder in SUBSET_FOLDER.items():
        where = root / folder
        if not where.is_dir():
            if subset == "a":
                raise SystemExit(
                    f"{where}: no such folder. --data-root is the folder holding "
                    "HyReCo/ (subset A) and, when downloaded, HyReCo-Additional/"
                )
            continue
        out.append((subset, where / REF_STAIN, where / MOV_STAIN))
    return out


def prepare(a):
    there = subsets_there(a.data_root)
    if getattr(a, "count", False):
        print(sum(len(SUBSET_IDS[s]) for s, _, _ in there))
        return 0
    todo, missing = [], []
    for subset, ref_dir, mov_dir in there:
        ids, gone = subset_ids(subset, ref_dir, mov_dir)
        todo += [(subset, cid, ref_dir, mov_dir) for cid in ids]
        missing += [f"{subset}_{i}" for i in gone]
    complete = not missing
    if missing and not a.allow_incomplete:
        raise SystemExit(
            f"{len(missing)} pair(s) of the subsets there are not in both stain folders "
            f"({', '.join(missing[:8])}{', ...' if len(missing) > 8 else ''}). "
            "--allow-incomplete prepares what is there and marks every case so."
        )
    settings = make_settings(
        getattr(a, "proxy", None),
        getattr(a, "reduce", None),
        getattr(a, "pixel_size_um", None),
    )
    if a.index is not None:  # into the subsets' lists, whatever is on disk
        listed = [(s, i) for s, _, _ in there for i in SUBSET_IDS[s]]
        pick = listed[a.index : a.index + 1]
        todo = [t for t in todo if (t[0], t[1]) in pick]
        missing = [m for m in missing if tuple(m.split("_", 1)) in pick]
    from .. import losses

    for case_id in missing:  # --allow-incomplete: a loss all the same, recorded
        losses.write_prepare_dropped(a.cases, HYRECO, case_id, {
            "dataset": HYRECO, "patient_id": case_id, "case_id": case_id,
            "status": "DROPPED",
            "reason": "the pair's slides are not in both stain folders here "
            "(--allow-incomplete)"})  # fmt: skip
    bad = 0
    for subset, cid, ref_dir, mov_dir in todo:
        case_id = f"{subset}_{cid}"
        lost = {"dataset": HYRECO, "patient_id": case_id, "case_id": case_id}
        try:
            d = prepare_one(
                a.cases, subset, cid, ref_dir, mov_dir, settings,
                settings["pixel_size_um"], a.force, complete,
            )  # fmt: skip
            print(f"{HYRECO}/{case_id} -> {d}")
            lost = None
        except Refused as exc:
            bad += 1
            print(f"{HYRECO}/{case_id}: DROPPED {exc}")
            lost.update(status="DROPPED", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - a row of prepare_dropped.csv
            bad += 1
            print(f"{HYRECO}/{case_id}: FAILED {type(exc).__name__}: {exc}")
            lost.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        losses.write_prepare_dropped(a.cases, HYRECO, case_id, lost)
    subsets = "+".join(s.upper() for s, _, _ in there)
    print(
        f"{HYRECO}: {len(todo) - bad}/{len(todo)} pair(s) of subset(s) {subsets} "
        f"prepared under {a.cases}; proxy {settings['proxy']}, reduced by "
        f"{settings['reduce']}"
        + (f"; {bad} dropped (prepare_dropped.csv)" if bad else "")
        + (f"; {len(missing)} not here (prepare_dropped.csv)" if missing else "")
    )
    return 1 if todo and bad == len(todo) else 0


# ── a look at a prepared case ─────────────────────────────────────────────────
def _crop(tif_path, xy, half):
    """``(2 half, 2 half, 3)`` uint8 around ``xy``, white outside the slide, with a
    black-and-yellow cross whose centre pixel is the landmark's."""
    import tifffile

    x, y = (int(round(float(v))) for v in xy)
    out = np.full((2 * half, 2 * half, 3), 255, np.uint8)
    with tifffile.TiffFile(str(tif_path)) as tif:
        read, (h, w) = _open_rgb(tif, tif_path)
        y0, y1 = max(y - half, 0), min(y + half, h)
        x0, x1 = max(x - half, 0), min(x + half, w)
        if y1 > y0 and x1 > x0:
            out[y0 - y + half : y1 - y + half, x0 - x + half : x1 - x + half] = read(
                y0, y1
            )[:, x0:x1]
    arm = max(4, half // 8)
    for d in range(3, arm):  # the centre is left open: the nucleus stays visible
        for dy, dx in ((d, 0), (-d, 0), (0, d), (0, -d)):
            out[half + dy, half + dx] = (255, 220, 0) if d % 2 else (0, 0, 0)
    return out


def landmark_crops(case, dest, n=8, half=128):
    """Write ``<dest>/<dataset>_<id>_landmark<i>.tif`` for the first ``n`` landmarks:
    the reference crop and the moving crop side by side. Returns the paths."""
    import tifffile

    pts = landmark_io.read_truth(case)
    if pts is None or pts["ref"] is None:
        raise SystemExit(f"{case.dataset}/{case.case_id}: the case has no truth.npz")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (r, m) in enumerate(zip(pts["ref"][:n], pts["mov"][:n])):
        pair = np.concatenate(
            [_crop(case.ref_image, r, half), _crop(case.mov_image, m, half)], axis=1
        )
        path = dest / f"{case.dataset}_{case.case_id}_landmark{i}.tif"
        tifffile.imwrite(str(path), pair, photometric="rgb")
        out.append(path)
    return out


def main(argv=None):
    import argparse

    from ..cases import load_case

    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.regbench.datasets.hyreco",
        description="Crops of both slides around the first landmarks of a prepared "
        "case, to check by eye that the landmark files are read as they were meant.",
    )
    ap.add_argument("--cases", required=True)
    ap.add_argument("--dataset", default=HYRECO, choices=(HYRECO,))
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--half", type=int, default=128, help="half the crop side, px")
    a = ap.parse_args(argv)
    case = load_case(case_dir(a.cases, a.dataset, a.case_id))
    for path in landmark_crops(case, a.out, a.n, a.half):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
