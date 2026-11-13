"""IBEX stitched fluorescence cycles as multiplex-style cases.

Two sources, one cycle per file, read in the layout verified against the real files
(2026-10-14):

``Human_{Spleen,Liver,Kidney}_Panel{1..4}.ims`` (Zenodo 4632320 `[MEMORY]`)
    HDF5. ``DataSet/ResolutionLevel 0/TimePoint 0/Channel N/Data`` is ``(Z, Y, X)`` and
    PADDED: it is cropped to ``DataSetInfo/Image`` attributes ``X`` and ``Y`` (byte-char
    arrays). A channel's name is ``DataSetInfo/Channel N`` attribute ``Name`` (e.g.
    ``"Panel4 : Hoechst\\n\\n"``, kept whole, stripped). The voxel is
    ``(ExtMax - ExtMin) / X``. ``NumberOfChannels`` may exceed the channel groups
    present: the groups are counted. Human panels have Z = 1; a Z > 1 is max-projected.

``spleen_panel{1,2,3}.nrrd`` (mouse spleen, Zenodo 4304786 `[MEMORY]`)
    gzip NRRD, ``sizes: X Y Z C`` (X fastest), uint8; the voxel is the length of the
    first two ``space directions`` vectors; channel names are in the
    ``imaris_channels_information:=`` XML. Max-projected over Z.

**Memory (user ruling 2026-10-14, "super memory efficient").** A cycle is never held
whole. An ``.ims`` is read one output tile at a time by sliced HDF5 reads. An ``.nrrd`` is
decompressed sequentially, one Z plane of one channel at a time, into a running maximum
(one Y x X plane); each channel's projection is written out before the next channel is
started. The OME-TIFF goes through ``imageio.write_channel_tiles`` (one encoder thread).

The nuclear channel is the one whose name contains ``hoechst`` or ``jojo`` (any case); a
cycle with none is refused naming the channels it has. Channels keep the file's order and
the nuclear index is recorded (``ref_channel`` / ``mov_channel``), as ``multiplex`` rows
carry it.

**Cases.** ``Panel1`` / ``panel1`` is the reference of its group; every later panel is a
case ``<group>_P<n>``. Human tissues are separate groups (``Human_Spleen``, ...), the mouse
spleen is ``Mouse_Spleen``. Cases are written by ``multiplex.prepare_one`` as dataset
``ibex``: own tables, never pooled with the cohort.

**Pixel size.** From the file. The mouse ``.nrrd`` header is checked against the record
below (0.284 um); a header more than 2 % off it is refused. For the human ``.ims`` there is
NO independent record: only a non-finite or non-positive voxel (or an X voxel that
disagrees with the Y voxel by more than 2 %) is refused, and ``case.json`` says the pixel
size comes from the header alone. Every panel of a group must have its reference's pixel
size (2 %).
"""

from __future__ import annotations

import gzip
import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .. import imageio
from . import multiplex

DATASET = "ibex"
WRITE_TILE = 1024  # the OME-TIFF's tile
TOLERANCE = 0.02
NUCLEAR_WORDS = ("hoechst", "jojo")
# the independent record of the pixel size, per source file kind (um). None: there is none
RECORD_UM = {"nrrd": 0.284, "ims": None}
NO_RECORD = "no independent record for human panels; the header alone"

IMS_RE = re.compile(r"^(Human_[A-Za-z]+)_Panel(\d+)\.ims$")
NRRD_RE = re.compile(r"^spleen_panel(\d+)\.nrrd$", re.IGNORECASE)
MOUSE_GROUP = "Mouse_Spleen"
_DTYPES = {"uint8": "u1", "uint16": "u2"}


class Info(NamedTuple):
    kind: str  # "ims" | "nrrd"
    names: list
    nuclear: int
    hw: tuple
    z: int
    pixel_size_um: float
    pixel_size_source: str
    dtype: str
    n_channels: int
    candidates: list  # every nuclear-looking channel, in file order
    extra: dict  # reader details (the NRRD's data offset, byte order)


# ── shared checks ────────────────────────────────────────────────────────────
def nuclear_candidates(names):
    return [i for i, n in enumerate(names) if any(w in n.lower() for w in NUCLEAR_WORDS)]


def nuclear_index(names, src):
    """The channel whose name contains Hoechst or JOJO (case-insensitive); the first
    when several do. Refused, naming the channels found, when none does."""
    found = nuclear_candidates(names)
    if not found:
        raise ValueError(
            f"{src}: no nuclear channel (a name containing 'Hoechst' or 'JOJO'); "
            f"the channels are {list(names)}"
        )
    return found[0]


def _check_px(px, kind, src):
    if not math.isfinite(px) or px <= 0:
        raise ValueError(f"{src}: the header gives a pixel size of {px}")
    record = RECORD_UM[kind]
    if record is None:
        return NO_RECORD
    if abs(px - record) / record > TOLERANCE:
        raise ValueError(
            f"{src}: the header's pixel size {px:.4g} um is more than "
            f"{TOLERANCE:.0%} off the record, {record} um"
        )
    return f"header; agrees with the record {record} um"


def _agree(px_x, px_y, src):
    for p in (px_x, px_y):
        if not math.isfinite(p) or p <= 0:
            raise ValueError(f"{src}: the header gives a pixel size of {p}")
    if abs(px_x - px_y) / px_x > TOLERANCE:
        raise ValueError(
            f"{src}: the X voxel {px_x:.4g} um and the Y voxel {px_y:.4g} um differ "
            f"by more than {TOLERANCE:.0%}"
        )


# ── .ims ─────────────────────────────────────────────────────────────────────
def _text(attr):
    """An Imaris attribute (a byte-char array, bytes or str) as a stripped string."""
    if isinstance(attr, np.ndarray):
        attr = b"".join(bytes(x) for x in attr.ravel()) if attr.dtype.kind == "S" else attr.tolist()
    if isinstance(attr, (bytes, bytearray)):
        attr = bytes(attr).decode("utf-8", "replace")
    return str(attr).strip()


def _ims_channels(f):
    base = f["DataSet/ResolutionLevel 0/TimePoint 0"]
    n = 0
    while f"Channel {n}" in base:
        n += 1
    return n


def read_ims(path):
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        img = f["DataSetInfo/Image"].attrs
        try:
            x, y = int(float(_text(img["X"]))), int(float(_text(img["Y"])))
            ext = [float(_text(img[k])) for k in ("ExtMin0", "ExtMax0", "ExtMin1", "ExtMax1")]
        except KeyError as exc:
            raise ValueError(f"{path}: DataSetInfo/Image has no attribute {exc}") from None
        z = int(float(_text(img["Z"]))) if "Z" in img else 1
        n = _ims_channels(f)
        if n == 0:
            raise ValueError(f"{path}: no 'Channel N' group under TimePoint 0")
        names = [_text(f[f"DataSetInfo/Channel {i}"].attrs["Name"]) for i in range(n)]
        dtype = str(f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"].dtype)
    px_x, px_y = (ext[1] - ext[0]) / x, (ext[3] - ext[2]) / y
    _agree(px_x, px_y, path)
    source = "ims header ((ExtMax-ExtMin)/X); " + _check_px(px_x, "ims", path)
    cands = nuclear_candidates(names)
    return Info(
        "ims", names, nuclear_index(names, path), (y, x), z, px_x, source, dtype, n,
        cands, {},
    )


def _ims_tiles(path, info, tile):
    """Every output tile of every channel (row-major per channel), by sliced reads."""
    import h5py

    y, x = info.hw
    with h5py.File(path, "r") as f:
        base = f["DataSet/ResolutionLevel 0/TimePoint 0"]
        for c in range(info.n_channels):
            ds = base[f"Channel {c}/Data"]
            if ds.shape[1] < y or ds.shape[2] < x:
                raise ValueError(
                    f"{path}: channel {c}'s data {ds.shape[1:]} is smaller than the "
                    f"declared image {(y, x)}"
                )
            for y0 in range(0, y, tile):
                for x0 in range(0, x, tile):
                    y1, x1 = min(y0 + tile, y), min(x0 + tile, x)
                    out = ds[0, y0:y1, x0:x1]
                    for zi in range(1, ds.shape[0]):
                        np.maximum(out, ds[zi, y0:y1, x0:x1], out=out)
                    yield out


# ── .nrrd ────────────────────────────────────────────────────────────────────
def _nrrd_header(path):
    fields, end = {}, 0
    with open(path, "rb") as fh:
        if not fh.readline().startswith(b"NRRD"):
            raise ValueError(f"{path}: not a NRRD file")
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: the header does not end")
            line = line.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                break
            if line.startswith("#"):
                continue
            if ":=" in line:
                k, v = line.split(":=", 1)
            elif ":" in line:
                k, v = line.split(":", 1)
            else:
                continue
            fields[k.strip().lower()] = v.strip()
        end = fh.tell()
    return fields, end


def _xml_names(xml):
    """The channel names of an ``imaris_channels_information`` XML: for each element
    whose tag contains 'channel', its ``Name`` attribute or ``Name`` child, in document
    order. The element layout is not verified against a real file: a count that does not
    equal the number of channels is refused by the caller."""
    import xml.etree.ElementTree as ET

    xml = xml.replace("\\n", "\n")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        root = ET.fromstring(f"<root>{xml}</root>")
    out = []
    for el in root.iter():
        if "channel" not in el.tag.lower():
            continue
        name = next((v for k, v in el.attrib.items() if k.lower() == "name"), None)
        if name is None:
            kid = next((k for k in el if k.tag.lower() == "name"), None)
            name = kid.text if kid is not None else None
        if name is not None and name.strip():
            out.append(name.strip())
    return out


def read_nrrd(path):
    path = Path(path)
    h, offset = _nrrd_header(path)
    if h.get("encoding", "").lower() not in ("gzip", "gz"):
        raise ValueError(f"{path}: encoding {h.get('encoding')!r}; only gzip is read")
    if h.get("type") not in _DTYPES:
        raise ValueError(f"{path}: type {h.get('type')!r}; only uint8 / uint16 are read")
    try:
        sizes = [int(v) for v in h["sizes"].split()]
    except (KeyError, ValueError):
        raise ValueError(f"{path}: no readable 'sizes'") from None
    if h.get("dimension") != "4" or len(sizes) != 4:
        raise ValueError(f"{path}: expected 4 dimensions (XYZC), got sizes {sizes}")
    x, y, z, c = sizes
    vecs = re.findall(r"\(([^)]*)\)", h.get("space directions", ""))
    if len(vecs) < 2:
        raise ValueError(f"{path}: no 'space directions' to take the pixel size from")
    px = [math.sqrt(sum(float(t) ** 2 for t in v.split(","))) for v in vecs[:2]]
    _agree(px[0], px[1], path)
    source = "nrrd header (space directions); " + _check_px(px[0], "nrrd", path)
    names = _xml_names(h.get("imaris_channels_information", ""))
    if len(names) != c:
        raise ValueError(
            f"{path}: {len(names)} channel names in imaris_channels_information for "
            f"{c} channels"
        )
    order = ">" if h.get("endian", "little").lower() == "big" else "<"
    return Info(
        "nrrd", names, nuclear_index(names, path), (y, x), z, px[0], source,
        h["type"], c, nuclear_candidates(names),
        {"offset": offset, "byteorder": order},
    )


def _nrrd_tiles(path, info, tile):
    """Every output tile of every channel: ONE channel's projection is built from its Z
    planes read in turn (a running maximum), written out, and released before the next."""
    y, x = info.hw
    dt = np.dtype(info.extra["byteorder"] + _DTYPES[info.dtype])
    nbytes = y * x * dt.itemsize
    plane = np.empty((y, x), dt)
    view = memoryview(plane.reshape(-1).view(np.uint8))
    with open(path, "rb") as fh:
        fh.seek(info.extra["offset"])
        with gzip.GzipFile(fileobj=fh, mode="rb") as gz:
            for c in range(info.n_channels):
                proj = None
                for zi in range(info.z):
                    got = 0
                    try:
                        while got < nbytes:
                            n = gz.readinto(view[got:])
                            if not n:
                                break
                            got += n
                    except (EOFError, OSError) as exc:
                        raise ValueError(f"{path}: the gzip stream is truncated ({exc})") from None
                    if got < nbytes:
                        raise ValueError(
                            f"{path}: the data ended in channel {c}, plane {zi} "
                            f"({got} of {nbytes} bytes): truncated"
                        )
                    if proj is None:
                        proj = plane.copy()
                    else:
                        np.maximum(proj, plane, out=proj)
                for y0 in range(0, y, tile):
                    for x0 in range(0, x, tile):
                        yield proj[y0 : y0 + tile, x0 : x0 + tile]
                del proj


# ── one cycle → one OME-TIFF ─────────────────────────────────────────────────
def read_info(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".ims":
        return read_ims(path)
    if suffix == ".nrrd":
        return read_nrrd(path)
    raise ValueError(f"{path}: not an .ims or .nrrd file")


def convert_cycle(src, dst, tile=WRITE_TILE):
    """Write ``src`` (one IBEX cycle) as an OME-TIFF with named channels, in the file's
    channel order, cropped and Z-projected, streaming (module docstring). Returns its
    :class:`Info`. A refusal leaves no ``dst``."""
    info = read_info(src)
    gen = (_ims_tiles if info.kind == "ims" else _nrrd_tiles)(src, info, tile)
    imageio.write_channel_tiles(
        dst, info.hw, np.dtype(info.dtype), gen, tile, info.names, info.pixel_size_um
    )
    return info


def names_of(path):
    return imageio.channel_names(path)


# ── the files, the groups, the rows ──────────────────────────────────────────
def discover(source):
    """``{group: {panel number: path}}`` of the IBEX files directly under ``source``."""
    found: dict = {}
    for p in sorted(Path(source).iterdir()):
        if m := IMS_RE.match(p.name):
            found.setdefault(m.group(1), {})[int(m.group(2))] = p
        elif m := NRRD_RE.match(p.name):
            found.setdefault(MOUSE_GROUP, {})[int(m.group(1))] = p
    return {g: dict(sorted(v.items())) for g, v in sorted(found.items())}


def rows(found, cache, tile=WRITE_TILE, force=False):
    """The ``multiplex`` rows: panel 1 against each later panel, per group. Converts each
    cycle once into ``<cache>/ome/``."""
    out_dir = Path(cache) / "ome"
    out = []
    for group, panels in found.items():
        if 1 not in panels:
            raise ValueError(f"{group}: no panel 1 (the reference); has {sorted(panels)}")
        conv = {}
        for n, src in panels.items():
            dst = out_dir / f"{group}_P{n}.ome.tif"
            info = read_info(src)  # the refusals first: nothing is written for a bad file
            if force or not dst.exists():
                convert_cycle(src, dst, tile)
            conv[n] = (dst, info)
        ref, ref_info = conv[1]
        for n, (dst, info) in conv.items():
            if n == 1:
                continue
            if abs(info.pixel_size_um - ref_info.pixel_size_um) / ref_info.pixel_size_um > TOLERANCE:
                raise ValueError(
                    f"{group}: panel {n}'s pixel size {info.pixel_size_um:.4g} um is more "
                    f"than {TOLERANCE:.0%} off panel 1's {ref_info.pixel_size_um:.4g} um"
                )
            out.append(
                {
                    "case_id": f"{group}_P{n}",
                    "reference": str(ref),
                    "moving": str(dst),
                    "group": group,
                    "ref_channel": str(ref_info.nuclear),
                    "mov_channel": str(info.nuclear),
                    "channels_ref": multiplex.CHANNEL_SEP.join(ref_info.names),
                    "channels_mov": multiplex.CHANNEL_SEP.join(info.names),
                    "pixel_size_um": repr(ref_info.pixel_size_um),
                    "pixel_size_source": ref_info.pixel_size_source,
                }
            )
    return out


def prepare(a):
    found = discover(a.source)
    if getattr(a, "groups", None):
        unknown = sorted(set(a.groups) - set(found))
        if unknown:
            raise SystemExit(f"--groups {unknown}: not in {a.source} ({sorted(found)})")
        found = {g: v for g, v in found.items() if g in a.groups}
    if not found:
        raise SystemExit(f"{a.source}: no IBEX .ims / .nrrd files")
    cache = Path(a.cases) / "_cache" / DATASET
    todo = []
    for group, panels in found.items():
        try:
            todo += rows({group: panels}, cache, force=a.force)
        except Exception as exc:
            print(f"{DATASET}/{group}: FAILED {type(exc).__name__}: {exc}")
    if a.index is not None:
        todo = todo[a.index : a.index + 1]
    bad = 0
    for row in todo:
        try:
            d = multiplex.prepare_one(a.cases, row, a, dataset=DATASET)
            print(f"{DATASET}/{row['case_id']} -> {d}")
        except Exception as exc:
            bad += 1
            print(f"{DATASET}/{row['case_id']}: FAILED {type(exc).__name__}: {exc}")
    return 1 if not todo or bad == len(todo) else 0
