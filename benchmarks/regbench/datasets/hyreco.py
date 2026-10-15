"""HyReCo, the external landmark set: re-stained H&E / PHH3 pairs as benchmark cases.

HyReCo (van der Laak, Lotz, Weiss, Heldmann, IEEE DataPort 2021, doi:10.21227/pzj5-bs61;
reference publication Lotz et al. 2023, J Med Imaging 10(6):067501,
doi:10.1117/1.JMI.10.6.067501) is the one public set found where the SAME section was
imaged twice (H&E, bleached, re-stained PHH3, scanned again) and annotated by hand.
Treated as CC BY-SA 4.0 (page title and paper; the DataCite record says CC BY 4.0).

**Two datasets, one rule.** ``prepare hyreco --subset a`` writes ``hyreco_a`` and
``--subset b`` writes ``hyreco_b``:

``hyreco_a``  subset A's nine re-stained pairs: the TUNING set. Every setting this module
              has (``--proxy``, ``--reduce``) is chosen here.
``hyreco_b``  subset B ("HyReCo-Additional"), 54 re-stained pairs: the REPORT set.
              Nothing is chosen on it. ``--subset b`` takes no setting: it copies the
              settings of the prepared ``hyreco_a`` cases (``--settings-from``, all nine,
              one settings hash), refuses ``--proxy`` / ``--reduce``, and refuses to
              prepare beside ``hyreco_b`` cases that carry other settings.

A folder is accepted for a subset only when its case ids are that subset's (the lists
below, from the dataset page), so subset B's folder cannot be prepared as the tuning set
and subset A's cannot be reported. Four ids (29, 108, 361, 464) are in both lists; whether
they are the same slides is not stated by any source read. They are prepared in
``hyreco_b`` and left out of its report set (``extra["report_set"]`` False) unless
``--keep-shared-ids`` is given after the files have shown they are different slides; the
pooled line over all 54 pairs (``regbench.landmarks``) includes them either way.

**Roles.** H&E is the reference (scanned first, like cycle 1), PHH3 the moving slide; each
pair is its own patient (``group = case_id``). ``truth_ref`` / ``truth_mov`` are the
landmarks of the two slides in full-resolution pixels, paired by line.

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
                          haematoxylin counterstain, which no source read states: to be
                          compared with luminance on subset A only.
``--reduce 2``            the proxy block-averaged 2 x 2 (0.48 um per pixel); 1 = native.
                          Decided by a pilot on subset A. Landmarks and ``ref_hw`` stay in
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

**Assumed, because no file was opened; each is checked on the files and fails loudly:**

1. *The first TIFF series' first level is the full-resolution image, RGB, uint8.*
   Anything else refuses the case (``_open_rgb``).
2. *Landmarks are paired by line number.* The page implies it and does not say it. So the
   two CSVs of a pair must have the same number of points, or the case is refused (never
   truncated), and a line that is not ``x,y[,z]`` refuses the case wherever it is, except
   one header line at the top. A blank line between points is refused too: it could be a
   missing landmark. More than three columns (an id column, say) is refused: this
   module pairs by position, and a file with ids must be paired by them.
3. *Pixel = millimetres / pixel size*, with no half-pixel term. Whether "upper left
   corner" is the corner of the first pixel or its centre is not stated; the same
   convention on both slides moves both landmarks of a pair by the same half pixel.
4. *The pixel size.* The conversion needs more than the paper's two digits: a landmark
   50 mm from the origin moves by 2000 px per 1 % of pixel size. So the FILE's value is
   used when it has one (resolution tags in inches or centimetres, or an ``MPP = `` field
   in the image description), and it must agree with the paper's 0.24 within 2 %, else
   the case is refused. A file with no value falls back to 0.24 exactly and says so
   (``pixel_size_source``; the summary repeats it, because two digits are then all the
   millimetre conversion rests on). ``--pixel-size-um`` overrides both, under the same
   2 % check. The two slides of a pair must agree within 0.1 %.
5. *Every landmark lies inside its image* after conversion, or the case is refused: a
   landmark outside means the pixel size or the origin is wrong.

Never run on the real files. Look at the first prepared case before any array is
submitted: ``python -m benchmarks.regbench.datasets.hyreco --cases C --dataset hyreco_a
--case-id 29 --out DIR`` writes, for the first landmarks of the pair, the crop of each
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

from ..cases import Case, case_dir, list_cases, write_case
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
DATASET = {"a": "hyreco_a", "b": "hyreco_b"}
TUNING, REPORT = DATASET["a"], DATASET["b"]
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
SETTINGS_VERSION = 2  # 2: pixel_size_um and keep_shared_ids joined the hash

LICENCE = "CC BY-SA 4.0 (dataset page and paper; DataCite says CC BY 4.0)"
CITATION = (
    "van der Laak, Lotz, Weiss, Heldmann, HyReCo, IEEE DataPort 2021, "
    "doi:10.21227/pzj5-bs61; Lotz et al. 2023, doi:10.1117/1.JMI.10.6.067501"
)


class Refused(ValueError):
    """A case this module will not prepare, with the reason."""


# ── settings ──────────────────────────────────────────────────────────────────
def make_settings(proxy=None, reduce=None, pixel_size_um=None, keep_shared_ids=False):
    """Every choice that reaches a result: the proxy, its reduction, a stated pixel size
    (None: the file's own) and whether subset B keeps the four ids subset A also has.
    Subset B takes the tuning set's, unchanged (``_settings_for``)."""
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
        "keep_shared_ids": bool(keep_shared_ids),
    }


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def tuned_settings(cases_root):
    """The settings of the prepared ``hyreco_a`` cases under ``cases_root``: all nine
    pairs, one settings hash. This is the only way a setting reaches subset B."""
    cases = list_cases(cases_root, TUNING)
    have = {c.case_id for c in cases}
    missing = [i for i in SUBSET_A_IDS if i not in have]
    if missing:
        raise SystemExit(
            f"{cases_root}: {TUNING} is not fully prepared (missing "
            f"{', '.join(missing)}). Subset B's settings are those of the nine "
            "subset-A pairs: prepare `hyreco --subset a` first."
        )
    hashes = {c.extra.get("settings_sha256") for c in cases}
    if len(hashes) != 1 or None in hashes:
        raise SystemExit(
            f"{cases_root}: the {TUNING} cases carry {len(hashes)} different settings; "
            "prepare the tuning set under one setting per cases folder"
        )
    settings = dict(cases[0].extra["settings"])
    if settings.get("version") != SETTINGS_VERSION:
        raise SystemExit(
            f"{cases_root}: the {TUNING} cases were prepared with settings version "
            f"{settings.get('version')}, this code writes {SETTINGS_VERSION} (the hash "
            "now covers the pixel size and the shared-ids flag): prepare them again"
        )
    if settings_hash(settings) != hashes.pop():
        raise SystemExit(f"{cases_root}: {TUNING} settings do not match their hash")
    return settings


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
    """``(um per pixel, source, what the file states or None)`` of one slide (module
    docstring, assumption 4)."""
    stated = file_pixel_size_um(path)
    if given:
        value, source = float(given), "command line (--pixel-size-um)"
    elif stated:
        value, source = stated, "file"
    else:
        value = PIXEL_SIZE_UM
        source = "paper, two digits (the file states none)"
    if abs(value - PIXEL_SIZE_UM) > PIXEL_SIZE_TOLERANCE * PIXEL_SIZE_UM:
        raise Refused(
            f"{path}: pixel size {value:.5f} um ({source}) is not within "
            f"{PIXEL_SIZE_TOLERANCE:.0%} of the published {PIXEL_SIZE_UM} um. Check the "
            "file; --pixel-size-um states a measured value"
        )
    return value, source, stated


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
    if xy.min() < -0.5 or xy[:, 0].max() > w - 0.5 or xy[:, 1].max() > h - 0.5:
        raise Refused(
            f"{csv}: landmarks span x {xy[:, 0].min():.0f}..{xy[:, 0].max():.0f}, "
            f"y {xy[:, 1].min():.0f}..{xy[:, 1].max():.0f} px on a {w} x {h} px image "
            f"at {px_um:.5f} um/px: the pixel size or the origin is not what this "
            "module assumes"
        )
    return xy, header, csv


def prepare_one(
    cases_root,
    subset,
    cid,
    ref_dir,
    mov_dir,
    settings,
    pixel_size_um=None,
    force=False,
    report_set=True,
    settings_from="",
    complete=True,
):
    dataset = DATASET[subset]
    d = case_dir(cases_root, dataset, cid)
    if (d / "case.json").exists() and not force:
        return d
    ref_t, mov_t = Path(ref_dir) / f"{cid}.tif", Path(mov_dir) / f"{cid}.tif"
    for p in (ref_t, mov_t, ref_t.with_suffix(".csv"), mov_t.with_suffix(".csv")):
        if not p.is_file():
            raise Refused(f"{p}: missing")
    f = int(settings["reduce"])
    tag = f"{settings['proxy']}.r{f}"
    cache = Path(cases_root) / "_cache" / dataset
    ref_n = cache / REF_STAIN / f"{cid}.{tag}.ome.tif"
    mov_n = cache / MOV_STAIN / f"{cid}.{tag}.ome.tif"
    # landmarks before the proxies: a refused pair costs no whole-slide read
    import tifffile

    hws = []
    for t in (ref_t, mov_t):
        with tifffile.TiffFile(str(t)) as tif:
            hws.append(_open_rgb(tif, t)[1])
    ref_px, ref_source, ref_stated = pixel_size(ref_t, pixel_size_um)
    mov_px, mov_source, mov_stated = pixel_size(mov_t, pixel_size_um)
    if abs(ref_px - mov_px) > PAIR_PIXEL_TOLERANCE * ref_px:
        raise Refused(
            f"{cid}: the reference is {ref_px:.5f} um/px and the moving slide "
            f"{mov_px:.5f}; one pixel size per pair is assumed"
        )
    ref_xy, ref_header, ref_csv = _landmarks_px(ref_t, hws[0], ref_px)
    mov_xy, mov_header, mov_csv = _landmarks_px(mov_t, hws[1], mov_px)
    if len(ref_xy) != len(mov_xy):
        raise Refused(
            f"{cid}: {len(ref_xy)} landmarks in {ref_csv.name} ({REF_STAIN}) and "
            f"{len(mov_xy)} in {mov_csv.name} ({MOV_STAIN}); they are paired by line, "
            "so unequal files are refused, not truncated"
        )
    _, ref_proxy_hw = write_proxy(ref_t, ref_n, settings, ref_px, force)
    _, mov_proxy_hw = write_proxy(mov_t, mov_n, settings, mov_px, force)
    case = Case(
        dataset=dataset,
        case_id=str(cid),
        group=str(cid),  # each pair is its own patient
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
            "role": "tuning" if subset == "a" else "report",
            "report_set": bool(report_set),
            "shared_id_with_subset_a": subset == "b" and str(cid) in SHARED_IDS,
            "subset_complete": bool(complete),
            "stain_ref": REF_STAIN,
            "stain_mov": MOV_STAIN,
            "n_landmarks": int(len(ref_xy)),
            "landmark_files": {"ref": str(ref_csv), "mov": str(mov_csv)},
            "landmark_header": {"ref": ref_header, "mov": mov_header},
            "landmark_pairing": "by line (assumed; the page implies it)",
            "landmark_origin": "px = mm / pixel size, no half-pixel term (assumed)",
            "pixel_size_source": ref_source,
            "pixel_size_file_um": {"ref": ref_stated, "mov": mov_stated},
            "pixel_size_mov_um": mov_px,
            "settings": dict(settings),
            "settings_sha256": settings_hash(settings),
            "settings_from": str(settings_from),
            "proxy_hw": {"ref": list(ref_proxy_hw), "mov": list(mov_proxy_hw)},
            "truth": "manual landmarks (Fraunhofer MEVIS)",
            "licence": LICENCE,
            "citation": CITATION,
        },
    )
    return write_case(cases_root, case, truth_mov=mov_xy, truth_ref=ref_xy)


# ── the subset ────────────────────────────────────────────────────────────────
def _stems(folder):
    return {p.stem for p in Path(folder).glob("*.tif")}


def subset_ids(subset, ref_dir, mov_dir, allow_incomplete=False):
    """The ids to prepare, in the page's list order, after checking that the two folders
    ARE this subset: stain folder names, no id of another subset, none missing."""
    ref_dir, mov_dir = Path(ref_dir), Path(mov_dir)
    for folder, stain in ((ref_dir, REF_STAIN), (mov_dir, MOV_STAIN)):
        if not folder.is_dir():
            raise SystemExit(f"{folder}: no such folder")
        if folder.name.upper() != stain:
            raise SystemExit(
                f"{folder}: expected the dataset's `{stain}` folder (H&E is the "
                "reference, PHH3 the moving slide). The consecutive pairs of subset A "
                "are not part of this module."
            )
    wanted = SUBSET_IDS[subset]
    found = _stems(ref_dir) | _stems(mov_dir)
    foreign = sorted(found - set(wanted), key=lambda s: (len(s), s))
    if foreign:
        other = "B" if subset == "a" else "A"
        raise SystemExit(
            f"{ref_dir.parent}: this is not subset {subset.upper()}: it holds "
            f"{len(foreign)} case(s) that subset does not have "
            f"({', '.join(foreign[:8])}{', ...' if len(foreign) > 8 else ''}). "
            f"Subset A is the tuning set and subset B the report set; a subset-{other} "
            "folder is never prepared under the other's name."
        )
    have = _stems(ref_dir) & _stems(mov_dir)
    missing = [i for i in wanted if i not in have]
    if missing and not allow_incomplete:
        raise SystemExit(
            f"subset {subset.upper()} has {len(wanted)} pairs; {len(missing)} are not "
            f"in both {ref_dir} and {mov_dir} ({', '.join(missing[:8])}"
            f"{', ...' if len(missing) > 8 else ''}). --allow-incomplete prepares what "
            "is there and marks every case so."
        )
    return [i for i in wanted if i in have], not missing


def _dirs(a):
    if a.ref_dir and a.mov_dir:
        return Path(a.ref_dir), Path(a.mov_dir)
    if a.data_root and not (a.ref_dir or a.mov_dir):
        return Path(a.data_root) / REF_STAIN, Path(a.data_root) / MOV_STAIN
    raise SystemExit(
        "give --data-root (a folder holding HE/ and PHH3/) or both --ref-dir and "
        "--mov-dir (subset B ships as two archives)"
    )


def _settings_for(a, subset):
    pixel = getattr(a, "pixel_size_um", None)
    keep = bool(getattr(a, "keep_shared_ids", False))
    if subset == "a":
        if a.settings_from:
            raise SystemExit("--settings-from belongs to --subset b")
        return make_settings(a.proxy, a.reduce, pixel, keep), ""
    if a.proxy is not None or a.reduce is not None:
        raise SystemExit(
            "--subset b takes no --proxy and no --reduce: nothing is chosen on subset "
            "B. Its settings are those of the prepared subset-A cases "
            "(--settings-from)."
        )
    if not a.settings_from:
        raise SystemExit(
            "--subset b needs --settings-from <cases folder holding the prepared "
            f"{TUNING} cases>: the settings chosen on subset A are the only ones "
            "subset B is prepared with"
        )
    settings = tuned_settings(a.settings_from)
    for key, asked in (("pixel_size_um", pixel), ("keep_shared_ids", keep)):
        if key == "pixel_size_um" and asked is not None:
            asked = float(asked)
        if asked != settings[key]:
            raise SystemExit(
                f"--subset b: {key} is {asked!r} here and {settings[key]!r} in the "
                f"tuning set ({a.settings_from}). Nothing is chosen on subset B: give "
                f"the tuning set's value, or prepare `--subset a` with {key} {asked!r} "
                "first and start the tuning again."
            )
    sha = settings_hash(settings)
    for case in list_cases(a.cases, REPORT):
        if case.extra.get("settings_sha256") != sha:
            raise SystemExit(
                f"{a.cases}: {REPORT}/{case.case_id} was prepared with other "
                f"settings ({case.extra.get('settings')}) than the tuning set's "
                f"({settings}). A second setting on subset B is tuning on it: remove "
                f"{Path(a.cases) / REPORT} and every result on it to start again."
            )
    return settings, str(Path(a.settings_from).resolve())


def prepare(a):
    subset = str(a.subset).lower()
    if subset not in DATASET:
        raise SystemExit(f"--subset {a.subset!r}: expected a or b")
    if getattr(a, "count", False):
        print(len(SUBSET_IDS[subset]))
        return 0
    ref_dir, mov_dir = _dirs(a)
    ids, complete = subset_ids(subset, ref_dir, mov_dir, a.allow_incomplete)
    settings, settings_from = _settings_for(a, subset)
    if a.index is not None:  # into the page's list, whatever is on disk
        wanted = SUBSET_IDS[subset]
        ids = [i for i in wanted[a.index : a.index + 1] if i in ids]
    dataset, bad = DATASET[subset], 0
    for cid in ids:
        report = subset == "b" and (
            cid not in SHARED_IDS or bool(settings["keep_shared_ids"])
        )
        try:
            d = prepare_one(
                a.cases,
                subset,
                cid,
                ref_dir,
                mov_dir,
                settings,
                settings["pixel_size_um"],
                a.force,
                report,
                settings_from,
                complete,
            )
            print(f"{dataset}/{cid} -> {d}")
        except Refused as exc:
            bad += 1
            print(f"{dataset}/{cid}: REFUSED {exc}")
    role = "tuning set" if subset == "a" else "report set"
    print(
        f"{dataset} ({role}): {len(ids) - bad}/{len(ids)} case(s) prepared under "
        f"{a.cases}; proxy {settings['proxy']}, reduced by {settings['reduce']}"
    )
    return 1 if bad else 0


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

    from ..cases import load_points

    points = load_points(case)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (r, m) in enumerate(zip(points["truth_ref"][:n], points["truth_mov"][:n])):
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
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET.values()))
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
