"""A stand-in for the ``valis`` package, for the adapters' tests where VALIS is not installed.

It does three things and nothing else:

* **records** every call the adapters and the pipeline's ``bin/register.py`` make into
  it (``CALLS``: constructor keywords, ``register``, ``register_micro``, every writer
  call), so a test can assert the exact sequence and arguments of a row;
* keeps VALIS's **geometry conventions**, transcribed from the PyPI sdists
  ``valis-wsi==1.0.0`` and ``==1.2.0`` ``[PARTIAL: the functions named below]``:
  ``warp_tools.get_warp_scaling_factors``, ``warp_xy_rigid``, ``warp_xy``, ``warp_xy_inv``,
  ``warp_xy_from_to``; ``Slide.get_crop_xywh``; ``Valis.get_aligned_slide_shape``;
* **draws** a registered slide the way ``warp_tools.warp_img`` is written to
  (``draw``): crop origin rounded as the configured version rounds it, the field scaled
  up and added to the pixel index, the rigid chain, then the moving slide sampled. That
  is an emulation of the writer written from the source, never the real one.

It does NOT register anything: ``Valis.register`` invents a similarity and a smooth field.
The selfcheck does not need a good registration, only a writer and a point path.

``install(...)`` puts the fake modules in ``sys.modules``; ``main()`` runs one scenario in
a fresh process (the pipeline's module rebinds ``multiprocessing.cpu_count``, so it is
never imported into the test process).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import types
from pathlib import Path

import numpy as np

CALLS = []
# What a test may set before ``Valis.register`` runs.
CONFIG = {
    "version": "1.2.0",
    "rounding": "ceil",  # the writer's crop origin: "ceil" (1.2.0) or "trunc" (1.0.0)
    "processed_dim": 200,  # the fake's rigid working size, whatever the caller asks
    "max_non_rigid": 2048,  # the first non-rigid pass's size when the caller names none
    "field_scale": 1.6,  # first non-rigid pass: field grid = this x the rigid canvas
    "micro_scale": 2.5,  # the micro pass's field grid
    "stored": False,  # first pass hands fields over as pyvips images (the tiler)
    "micro_stored": False,  # so does the micro pass
    "mutate_in_place": True,  # the micro pass also scribbles over the old arrays
    "field_order": 3,  # the writer's interpolation of the field
    "fail_register": False,
    "micro_raises": False,
    "skip_write": (),  # slide names the writer raises on
    "channel_names": None,
}
DEFAULTS = dict(CONFIG)


def record(name, **kw):
    CALLS.append({"call": name, **kw})


def calls(name):
    return [c for c in CALLS if c["call"] == name]


def _plain(v):
    if isinstance(v, type):
        return v.__name__
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return type(v).__name__


# ── pyvips ────────────────────────────────────────────────────────────────────
class VipsImage:
    """Enough of ``pyvips.Image`` for the adapters: a two-band field, or an xyz probe."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.height, self.width = self.array.shape[:2]
        self.bands = self.array.shape[2] if self.array.ndim == 3 else 1

    @classmethod
    def xyz(cls, width, height):
        yy, xx = np.mgrid[0:height, 0:width]
        return cls(np.dstack([xx, yy]).astype(float))

    def extract_area(self, left, top, width, height):
        left, top = int(left), int(top)
        return VipsImage(self.array[top : top + int(height), left : left + int(width)])

    def __call__(self, x, y):
        return [float(v) for v in np.atleast_1d(self.array[int(y), int(x)])]

    def stats(self):
        raise RuntimeError("the fake pyvips image has no stats")


def round_origin(xy, how=None):
    how = how or CONFIG["rounding"]
    xy = np.asarray(xy, dtype=float)
    return {"ceil": np.ceil, "trunc": np.trunc, "round": np.rint}[how](xy)


# ── warp_tools ────────────────────────────────────────────────────────────────
def vips2numpy(vi):
    record("vips2numpy")
    return np.asarray(vi.array)


def get_warp_scaling_factors(
    transformation_src_shape_rc=None,
    transformation_dst_shape_rc=None,
    src_shape_rc=None,
    dst_shape_rc=None,
    bk_dxdy=None,
    fwd_dxdy=None,
):
    """``warp_tools.py:1836-1939`` (1.0.0) / ``1995-2098`` (1.2.0), NumPy fields only."""
    field = bk_dxdy if bk_dxdy is not None else fwd_dxdy

    def arr(v):
        return None if v is None else np.array(v, dtype=float)

    src, tsrc = arr(src_shape_rc), arr(transformation_src_shape_rc)
    dst, tdst = arr(dst_shape_rc), arr(transformation_dst_shape_rc)
    src_sxy = None
    if tsrc is not None and src is not None and not np.all(tsrc == src):
        src_sxy = (src / tsrc)[::-1]
    disp = None if field is None else np.array(np.asarray(field)[0].shape, float)
    if tdst is None and disp is not None:
        tdst = disp
    if dst is None and tdst is not None:
        dst = tdst
    if disp is not None:
        if not np.all(tdst == disp):
            displacement_sxy = (disp / tdst)[::-1]
            dst_sxy = (dst / disp)[::-1]
        else:
            displacement_sxy = None
            dst_sxy = (dst / tdst)[::-1]
        if np.all(dst_sxy == 1):
            dst_sxy = None
    else:
        dst_sxy = displacement_sxy = None
        if tdst is not None and dst is not None and not np.all(dst == tdst):
            dst_sxy = (dst / tdst)[::-1]
    return src_sxy, dst_sxy, displacement_sxy, disp


def warp_xy_rigid(xy, inv_matrix):
    """Points forward through a matrix that maps registered -> source."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    h = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(inv_matrix).T
    return h[:, :2] / h[:, 2:3]


def warp_xy_non_rigid(xy, dxdy, displacement_shape_rc=None):
    """``warp_tools.py:1969-1992`` (1.2.0): a cubic spline through the field's samples,
    sample ``i`` at coordinate ``i``."""
    from scipy.interpolate import RectBivariateSpline

    dx, dy = np.asarray(dxdy[0], float), np.asarray(dxdy[1], float)
    rows, cols = np.arange(dx.shape[0]), np.arange(dx.shape[1])
    fx, fy = RectBivariateSpline(rows, cols, dx), RectBivariateSpline(rows, cols, dy)
    return np.column_stack(
        [
            xy[:, 0] + fx(xy[:, 1], xy[:, 0], grid=False),
            xy[:, 1] + fy(xy[:, 1], xy[:, 0], grid=False),
        ]
    )


def warp_xy(
    xy,
    M=None,
    transformation_src_shape_rc=None,
    transformation_dst_shape_rc=None,
    src_shape_rc=None,
    dst_shape_rc=None,
    bk_dxdy=None,
    fwd_dxdy=None,
    pt_buffer=100,
):
    """``_warp_xy_numpy``, ``warp_tools.py:2069-2161`` / ``2221-2304``."""
    if isinstance(bk_dxdy, VipsImage) or isinstance(fwd_dxdy, VipsImage):
        raise AssertionError("a pyvips field reached warp_xy: the per-point loop")
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    non_rigid = bk_dxdy is not None or fwd_dxdy is not None
    if M is None and not non_rigid:
        return xy
    src_sxy, dst_sxy, displacement_sxy, _ = get_warp_scaling_factors(
        transformation_src_shape_rc,
        transformation_dst_shape_rc,
        src_shape_rc,
        dst_shape_rc,
        bk_dxdy,
        fwd_dxdy,
    )
    pts = xy / src_sxy if src_sxy is not None else xy
    if M is not None:
        pts = warp_xy_rigid(pts, M)
        if not non_rigid:
            return pts * dst_sxy if dst_sxy is not None else pts
    if displacement_sxy is not None:
        pts = pts * displacement_sxy
    if fwd_dxdy is None:
        raise AssertionError("the fake does not invert a field, and nothing should ask")
    out = warp_xy_non_rigid(pts, fwd_dxdy)
    return out * dst_sxy if dst_sxy is not None else out


def warp_xy_inv(
    xy,
    M=None,
    transformation_src_shape_rc=None,
    transformation_dst_shape_rc=None,
    src_shape_rc=None,
    dst_shape_rc=None,
    bk_dxdy=None,
    fwd_dxdy=None,
):
    """``warp_tools.py:2235-2307`` / ``2377-2449``: registered canvas -> unwarped slide."""
    record("warp_xy_inv", has_field=bk_dxdy is not None, n=int(len(xy)))
    if isinstance(bk_dxdy, VipsImage):
        raise AssertionError("a pyvips field reached warp_xy_inv: the per-point loop")
    if fwd_dxdy is not None:
        raise AssertionError("a forward field would have to be inverted")
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if M is None and bk_dxdy is None:
        return xy
    src_sxy, dst_sxy, displacement_sxy, _ = get_warp_scaling_factors(
        transformation_src_shape_rc,
        transformation_dst_shape_rc,
        src_shape_rc,
        dst_shape_rc,
        bk_dxdy,
        fwd_dxdy,
    )
    pts = xy / dst_sxy if dst_sxy is not None else xy
    if bk_dxdy is not None:
        pts = warp_xy(pts, fwd_dxdy=bk_dxdy)
        if displacement_sxy is not None:
            pts = pts / displacement_sxy
    if M is not None:
        pts = warp_xy(pts, M=np.linalg.inv(M))
    return pts * src_sxy if src_sxy is not None else pts


def warp_xy_from_to(xy, **kw):
    """``warp_tools.py:2310-2399`` / ``2452-2541``."""
    record(
        "warp_xy_from_to",
        keywords=sorted(kw),
        to_bk_dxdy=id(kw.get("to_bk_dxdy")),
        has_field=kw.get("to_bk_dxdy") is not None,
    )
    names = (
        "M",
        "transformation_src_shape_rc",
        "transformation_dst_shape_rc",
        "src_shape_rc",
        "dst_shape_rc",
        "bk_dxdy",
        "fwd_dxdy",
    )
    reg = warp_xy(xy, **{n: kw.get(f"from_{n}") for n in names})
    return warp_xy_inv(reg, **{n: kw.get(f"to_{n}") for n in names})


def warp_img(img, M=None, bk_dxdy=None, bbox_xywh=None, **kw):
    """Only what the adapters' crop probe uses: no transform, a crop box, and the
    version's rounding of its origin (``np.ceil`` at 1.2.0 ``warp_tools.py:1118``)."""
    record("warp_img", bbox_xywh=[float(v) for v in bbox_xywh])
    if M is not None or bk_dxdy is not None or not isinstance(img, VipsImage):
        raise AssertionError("the fake warp_img is the crop probe only")
    x, y = round_origin(bbox_xywh[:2])
    return img.extract_area(x, y, bbox_xywh[2], bbox_xywh[3])


def get_inverse_field(bk_dxdy, n_inter=10):
    record("get_inverse_field")
    raise AssertionError("an iterative inverse was asked for")


# ── the emulated writer ───────────────────────────────────────────────────────
def canvas_points(slide, xy_out, non_rigid=True, rounding=None):
    """Where ``warp_img`` samples the moving slide for output pixels ``xy_out`` of a slide
    written with ``crop="reference"`` (module docstring)."""
    from scipy.ndimage import map_coordinates

    reg = slide.val_obj
    ref = reg.get_ref_slide()
    ref_full = np.asarray(ref.slide_dimensions_wh[0], float)  # (W, H)
    sxy = ref_full / np.asarray(ref.processed_img_shape_rc, float)[::-1]
    origin = round_origin(-ref.M[:2, 2] * sxy, rounding)
    canvas = np.asarray(slide.aligned_slide_shape_rc, float)[::-1]  # (W, H)
    x = np.asarray(xy_out, float).reshape(-1, 2) + origin
    field = slide.bk_dxdy if non_rigid else None
    if isinstance(field, VipsImage):
        field = np.array([field.array[..., 0], field.array[..., 1]])
    if field is not None:
        k = canvas / np.asarray(field[0].shape, float)[::-1]
        at = [x[:, 1] / k[1], x[:, 0] / k[0]]
        order = CONFIG["field_order"]
        d = np.column_stack(
            [
                map_coordinates(np.asarray(f, float), at, order=order, mode="nearest")
                for f in field
            ]
        )
        x = x + k * d
    r = x * (np.asarray(slide.reg_img_shape_rc, float)[::-1] / canvas)
    src = np.column_stack([r, np.ones(len(r))]) @ np.asarray(slide.M, float).T
    full = np.asarray(slide.slide_dimensions_wh[0], float)
    return src[:, :2] * (full / np.asarray(slide.processed_img_shape_rc, float)[::-1])


def draw(slide, planes, non_rigid=True, rounding=None, order=1):
    """Every plane of the moving slide resampled onto the reference grid."""
    from scipy.ndimage import map_coordinates

    ref = slide.val_obj.get_ref_slide()
    w, h = (int(v) for v in ref.slide_dimensions_wh[0])
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    p = canvas_points(
        slide, np.column_stack([xx.ravel(), yy.ravel()]), non_rigid, rounding
    )
    at = [p[:, 1].reshape(h, w), p[:, 0].reshape(h, w)]
    return np.asarray(
        [
            np.rint(
                np.clip(map_coordinates(c.astype(float), at, order=order), 0, 65535)
            )
            for c in planes
        ],
        np.uint16,
    )


# ── registration ──────────────────────────────────────────────────────────────
def _name(path):
    name = os.path.basename(str(path))
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _smooth_field(shape_rc, amp, seed):
    """A smooth field in its own pixels: a few px, slowly varying."""
    h, w = shape_rc
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u, v = xx / max(w - 1, 1), yy / max(h - 1, 1)
    dx = amp * np.sin(2 * math.pi * (0.8 * u + 0.3 * v) + seed)
    dy = amp * np.cos(2 * math.pi * (0.5 * u - 0.7 * v) + 0.7 * seed)
    return np.array([dx, dy])


class _Metadata:
    def __init__(self, names):
        self.channel_names = list(names)
        self.n_channels = len(names)


class _Reader:
    def __init__(self, names):
        self.metadata = _Metadata(names)

    def get_channel_index(self, channel):
        hits = [
            i
            for i, n in enumerate(self.metadata.channel_names)
            if re.search(str(channel).lower(), str(n).lower())
        ]
        return hits[0] if len(hits) == 1 else 0


class ErrorTable:
    """Stands in for the summary DataFrame ``register`` returns."""

    def to_csv(self, path, index=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("from,to\n")

    def __repr__(self):
        return "<fake error table>"


class Slide:
    def __init__(self, src_f, val_obj):
        import tifffile

        self.src_f = str(src_f)
        self.name = _name(src_f)
        self.val_obj = val_obj
        with tifffile.TiffFile(self.src_f) as tif:
            shape = tif.series[0].shape
            page = tif.pages[0]
            # Bio-Formats' isRGB, which both sdists' BioFormatsSlideReader._check_rgb
            # return (1.0.0 slide_io.py:1161-1171, 1.2.0 :1513-1523); taken here as an
            # image of three samples per pixel (Bio-Formats' definition, more than one
            # sample per pixel, is `[MEMORY]`). Three separate channels are not RGB.
            self.is_rgb = bool(
                page.samplesperpixel == 3 and page.photometric.name == "RGB"
            )
        # slide_io guess_image_type (1.0.0 :666-684, 1.2.0 :1017-1035): RGB = brightfield
        self.img_type = "brightfield" if self.is_rgb else "fluorescence"
        h, w = shape[:2] if self.is_rgb else shape[-2:]
        self.slide_dimensions_wh = np.array([[w, h]])
        n = 3 if self.is_rgb else shape[0] if len(shape) == 3 else 1
        names = CONFIG["channel_names"]
        if names is None:
            names = ["DAPI", "dots"][:n]
        if len(names) != n:  # a slide of other channels: the names in its header
            from benchmarks.regbench import imageio

            try:
                names = imageio.channel_names(self.src_f)
            except Exception:
                names = []
            if len(names) != n:
                names = ["R", "G", "B"] if self.is_rgb else [f"C{i}" for i in range(n)]
        self.reader = _Reader(names)
        self.M = None
        self.processed_img_shape_rc = None
        self.reg_img_shape_rc = None
        self.aligned_slide_shape_rc = None
        self.stored_dxdy = False
        self._bk = None
        self._fwd = None
        self.bg_color = [0, 0]

    # the fields are read through properties, as in 1.2.0 (registration.py:663-728)
    @property
    def bk_dxdy(self):
        record("get_bk_dxdy", slide=self.name, generation=self.val_obj.generation)
        if self._bk is None:
            return None
        if self.stored_dxdy:
            return VipsImage(np.dstack(self._bk))
        return self._bk

    @bk_dxdy.setter
    def bk_dxdy(self, value):
        self._bk = value

    @property
    def fwd_dxdy(self):
        if self._fwd is None:
            return None
        if self.stored_dxdy:
            return VipsImage(np.dstack(self._fwd))
        return self._fwd

    @fwd_dxdy.setter
    def fwd_dxdy(self, value):
        self._fwd = value

    def get_crop_xywh(self, crop, out_shape_rc=None):
        """``registration.py:461-497, 533-566`` / ``502-538, 574-607``, crop="reference"."""
        if crop != "reference":
            raise AssertionError(f"crop={crop!r}: only 'reference' is expected")
        ref = self.val_obj.get_ref_slide()
        out = np.asarray(out_shape_rc, float)
        sxy = out[::-1] / np.asarray(ref.processed_img_shape_rc, float)[::-1]
        return (*(-ref.M[0:2, 2] * sxy), *out[::-1]), None

    def warp_and_save_slide(
        self,
        dst_f,
        level=0,
        non_rigid=True,
        crop=True,
        src_f=None,
        channel_names=None,
        colormap=None,
        interp_method="bicubic",
        **kw,
    ):
        import tifffile

        from benchmarks.regbench import imageio

        record(
            "warp_and_save_slide",
            slide=self.name,
            dst_f=str(dst_f),
            level=level,
            non_rigid=non_rigid,
            crop=crop,
            src_f=None if src_f is None else str(src_f),
            interp_method=interp_method,
            generation=self.val_obj.generation,
            extra=sorted(kw),
        )
        if self.name in CONFIG["skip_write"]:
            raise RuntimeError(f"the fake writer refuses {self.name}")
        crop = self.val_obj.crop if crop is True else crop
        if crop != "reference":
            raise AssertionError(f"crop={crop!r}: only 'reference' is expected")
        planes = tifffile.imread(src_f or self.src_f)
        if self.is_rgb:  # (H, W, 3) samples: drawn as three planes
            planes = np.moveaxis(planes, -1, 0)
        planes = planes[None] if planes.ndim == 2 else planes
        if self is self.val_obj.get_ref_slide():
            out = planes  # written as it is (1.2.0); the benchmark never reads it
        else:
            out = draw(self, planes, non_rigid=non_rigid)
        names = self.reader.metadata.channel_names[: len(out)]
        Path(dst_f).parent.mkdir(parents=True, exist_ok=True)
        imageio.write_channels(dst_f, out, names)


class Valis:
    def __init__(self, src_dir, dst_dir, **kwargs):
        record(
            "Valis", src_dir=str(src_dir), dst_dir=str(dst_dir), kwargs=_plain(kwargs)
        )
        self.src_dir, self.dst_dir = str(src_dir), str(dst_dir)
        self.kwargs = kwargs
        self.name = os.path.basename(os.path.normpath(self.src_dir))
        self.data_dir = os.path.join(self.dst_dir, self.name, "data")
        files = kwargs.get("img_list")
        if files is None:
            files = sorted(
                os.path.join(self.src_dir, f)
                for f in os.listdir(self.src_dir)
                if f.lower().endswith((".tif", ".tiff"))
            )
        self.original_img_list = sorted(str(f) for f in files)
        self.reference_img_f = kwargs.get("reference_img_f")
        self.crop = kwargs.get("crop") or (
            "reference" if self.reference_img_f else "overlap"
        )
        self.max_non_rigid_registration_dim_px = kwargs.get(
            "max_non_rigid_registration_dim_px", CONFIG["max_non_rigid"]
        )
        self.slide_dict = {}
        self.generation = 0
        self._non_rigid_bbox = None
        self._full_displacement_shape_rc = None
        self.error_df = None

    def get_ref_slide(self):
        return self.slide_dict[_name(self.reference_img_f)]

    def get_slide(self, src_f):
        return self.slide_dict.get(_name(src_f))

    def get_aligned_slide_shape(self, level):
        """``registration.py:4414-4446`` / ``5065-5097`` at level 0."""
        ref = self.get_ref_slide()
        s_rc = np.asarray(ref.slide_dimensions_wh[0], float)[::-1] / np.asarray(
            ref.processed_img_shape_rc, float
        )
        return np.ceil(np.asarray(ref.reg_img_shape_rc) * s_rc).astype(int)

    def _field_shape(self, scale):
        ref = self.get_ref_slide()
        return tuple(np.ceil(np.asarray(ref.reg_img_shape_rc) * scale).astype(int))

    def register(self, **kw):
        record("register", kwargs=_plain(kw))
        if CONFIG["fail_register"]:
            # as VALIS does: the exception goes to valtils.print_warning (looked up on
            # the module at call time) and (None, None, None) comes back
            text = CONFIG["fail_register"]
            exc = RuntimeError(text if isinstance(text, str) else "the fake fails")
            valtils = sys.modules["valis.valtils"]
            if CONFIG["version"] == "1.0.0":
                valtils.print_warning(exc)
            else:
                valtils.print_warning(exc, rgb="red", traceback_msg="Traceback (fake)")
            return None, None, None
        for f in self.original_img_list:
            self.slide_dict[_name(f)] = Slide(f, self)
        ref = self.get_ref_slide()
        dim = CONFIG["processed_dim"]
        procs = {}
        for name, s in self.slide_dict.items():
            w, h = (int(v) for v in s.slide_dimensions_wh[0])
            f = dim / max(w, h)
            procs[name] = (int(round(h * f)), int(round(w * f)))
        t = np.array([7.3, 5.6])  # where the reference sits on the canvas, processed px
        reg_rc = (
            max(p[0] for p in procs.values()) + 19,
            max(p[1] for p in procs.values()) + 23,
        )
        micro_rigid = self.kwargs.get("micro_rigid_registrar_cls") is not None
        for i, (name, s) in enumerate(self.slide_dict.items()):
            s.processed_img_shape_rc = procs[name]
            s.reg_img_shape_rc = reg_rc
            if s is ref:
                s.M = np.array([[1.0, 0.0, -t[0]], [0.0, 1.0, -t[1]], [0.0, 0.0, 1.0]])
                continue
            th = math.radians(1.2 + 0.4 * i + (0.3 if micro_rigid else 0.0))
            c, sn = math.cos(th), math.sin(th)
            centre = np.array([reg_rc[1], reg_rc[0]]) / 2.0
            rot = np.array([[c, -sn], [sn, c]])
            b = centre - rot @ centre - t + np.array([1.7, -2.4])
            s.M = np.array([[c, -sn, b[0]], [sn, c, b[1]], [0.0, 0.0, 1.0]])
        aligned = self.get_aligned_slide_shape(0)
        shape = self._field_shape(CONFIG["field_scale"])
        self._full_displacement_shape_rc = shape
        self._non_rigid_bbox = [0, 0, shape[1], shape[0]]
        for i, s in enumerate(self.slide_dict.values()):
            s.aligned_slide_shape_rc = aligned
            s.stored_dxdy = bool(CONFIG["stored"])
            if s is ref:
                s._bk, s._fwd = np.zeros((2, *shape)), np.zeros((2, *shape))
            else:
                s._bk = _smooth_field(shape, 2.0, 1.0 + i)
                s._fwd = -s._bk
        self.error_df = ErrorTable()
        return object(), object(), self.error_df

    def measure_error(self):
        record("measure_error")
        return self.error_df

    def register_micro(self, **kw):
        record("register_micro", kwargs=_plain(kw), generation=self.generation)
        if CONFIG["micro_raises"]:
            text = CONFIG["micro_raises"]
            raise RuntimeError(
                text if isinstance(text, str) else "the fake micro pass fails"
            )
        size = kw.get("max_non_rigid_registration_dim_px")
        if CONFIG["version"] != "1.0.0":
            size = 4096 if size is None else size  # DEFAULT_MAX_MICRO_REG_SIZE
            if self.max_non_rigid_registration_dim_px >= size:  # 1.2.0 :4862-4871
                return None, self.error_df
        ref = self.get_ref_slide()
        shape = self._field_shape(CONFIG["micro_scale"])
        for i, s in enumerate(self.slide_dict.values()):
            if s is ref:  # skipped, as at 1.0.0 :4293 / 1.2.0 :4944
                continue
            old = s._bk
            s._bk = _smooth_field(shape, 3.0, 4.0 + i)
            s._fwd = -s._bk
            s.stored_dxdy = bool(CONFIG["micro_stored"])
            if CONFIG["mutate_in_place"]:
                old += 40.0  # a reference kept from before would now be wrong
        self._full_displacement_shape_rc = shape
        self._non_rigid_bbox = [0, 0, shape[1], shape[0]]
        self.generation += 1
        return object(), self.error_df

    def warp_and_save_slides(self, dst_dir, **kw):
        record(
            "warp_and_save_slides",
            dst_dir=str(dst_dir),
            kwargs=_plain(kw),
            generation=self.generation,
        )
        for s in self.slide_dict.values():
            s.warp_and_save_slide(
                dst_f=os.path.join(str(dst_dir), s.name + ".ome.tiff"),
                non_rigid=kw.get("non_rigid", True),
                crop=kw.get("crop", True),
                src_f=s.src_f,
            )


# ── the package ───────────────────────────────────────────────────────────────
class MicroRigidRegistrar:
    def align_slides(self, moving_slide, fixed_slide, *args, **kwargs):
        return None


class OpticalFlowWarper:
    pass


class SuperPointFD:
    def detect_and_compute(self, *a, **k):
        return None

    def detect_and_compute_sg(self, *a, **k):
        return None

    def compute(self, *a, **k):
        return None


class BioFormatsSlideReader:
    def slide2vips(self, level, series=None, xywh=None, *args, **kwargs):
        return None

    def slide2image(self, level, series=None, xywh=None, *args, **kwargs):
        return None


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def modules(version):
    """The fake ``valis`` package (and ``pyvips``, ``jpype``) as ``{name: module}``."""
    fd = _module(
        "valis.feature_detectors", MAX_FEATURES=20000, SuperPointFD=SuperPointFD
    )

    def filter_features(kp, desc, n_keep=20000):
        return kp, desc

    fd.filter_features = filter_features

    class SuperGlueMatcher:
        def __init__(self):
            self.config = {"superpoint": {"max_keypoints": fd.MAX_FEATURES}}

        def match_images(self, *a, **k):
            return None

    def slide_io_init_jvm(jar=None, mem_gb=10):
        record("slide_io.init_jvm", mem_gb=mem_gb)

    def init_jvm(jar=None, mem_gb=10):
        # both sdists pass the literal on: registration.init_jvm discards its argument
        record("init_jvm", mem_gb=mem_gb)
        slide_io_init_jvm(jar=None, mem_gb=10)

    def print_warning(msg, warning_type=None, rgb=None, traceback_msg=None):
        record("print_warning", msg=str(msg))

    def kill_jvm():
        record("kill_jvm")

    def get_ncpus_available():
        record("get_ncpus_available")
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 2

    mods = {
        "valis.registration": _module(
            "valis.registration",
            Valis=Valis,
            Slide=Slide,
            init_jvm=init_jvm,
            kill_jvm=kill_jvm,
        ),
        "valis.warp_tools": _module(
            "valis.warp_tools",
            vips2numpy=vips2numpy,
            get_warp_scaling_factors=get_warp_scaling_factors,
            warp_xy_rigid=warp_xy_rigid,
            warp_xy=warp_xy,
            warp_xy_inv=warp_xy_inv,
            warp_xy_from_to=warp_xy_from_to,
            warp_img=warp_img,
            get_inverse_field=get_inverse_field,
        ),
        "valis.valtils": _module(
            "valis.valtils",
            get_ncpus_available=get_ncpus_available,
            print_warning=print_warning,
        ),
        "valis.slide_io": _module(
            "valis.slide_io",
            BioFormatsSlideReader=BioFormatsSlideReader,
            init_jvm=slide_io_init_jvm,
        ),
        "valis.feature_detectors": fd,
        "valis.feature_matcher": _module(
            "valis.feature_matcher", SuperGlueMatcher=SuperGlueMatcher
        ),
        "valis.micro_rigid_registrar": _module(
            "valis.micro_rigid_registrar", MicroRigidRegistrar=MicroRigidRegistrar
        ),
        "valis.non_rigid_registrars": _module(
            "valis.non_rigid_registrars", OpticalFlowWarper=OpticalFlowWarper
        ),
    }
    pkg = _module("valis", __version__=version, __file__=__file__, __path__=[])
    for full, mod in mods.items():
        setattr(pkg, full.split(".", 1)[1], mod)
    return {
        "valis": pkg,
        **mods,
        "pyvips": _module("pyvips", Image=VipsImage),
        "jpype": _module("jpype", isJVMStarted=lambda: True),
    }


def configure(**config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown fake VALIS settings: {sorted(unknown)}")
    CONFIG.clear()
    CONFIG.update({**DEFAULTS, **config})
    del CALLS[:]


def install(setitem=None, **config):
    """Configure the fake and put it in ``sys.modules`` (through ``monkeypatch.setitem``
    when given, so a test leaves nothing behind)."""
    configure(**config)
    put = setitem or (lambda mapping, key, value: mapping.__setitem__(key, value))
    for name, mod in modules(CONFIG["version"]).items():
        put(sys.modules, name, mod)


def _torch_stub():
    """``torch`` for the pipeline's ``valis_config``, which only needs these three."""
    import contextlib

    threads = [os.cpu_count() or 1]
    return _module(
        "torch",
        no_grad=contextlib.nullcontext,
        set_num_threads=lambda n: threads.__setitem__(0, int(n)),
        get_num_threads=lambda: threads[0],
    )


def main(argv=None):
    """One scenario in this (fresh) process: ``fake_valis.py <request.json>``.

    The request names the method, the work folder, the fake's settings and the options
    (``rounds``: the moving rounds of the case, 1 for a pair, 2 for a patient);
    the answer (the selfcheck record, the calls, what VALIS's pools were left with) is
    written beside it as ``answer.json``."""
    import multiprocessing

    request = json.loads(Path((argv or sys.argv[1:])[0]).read_text())
    install(**request.get("fake", {}))
    sys.modules["torch"] = _torch_stub()
    for key, value in (request.get("env") or {}).items():
        os.environ[key] = str(value)
    if request.get("mode") == "group":
        return _group(request)
    from benchmarks.regbench import selfcheck

    rec = selfcheck.run(
        request["method"],
        request["work"],
        request.get("opts"),
        label=request.get("label"),
        sizes=[(tuple(request["size"]), *([None] * int(request.get("rounds", 1))))],
        in_process=True,
        keep_work=True,
    )
    answer = {
        "selfcheck": rec,
        "calls": CALLS,
        "cpu_count": multiprocessing.cpu_count(),
    }
    Path(request["work"], "answer.json").write_text(json.dumps(answer, default=str))
    return 0


# a point the maps of ``run_group`` are evaluated at, so a row can say it has one
PROBE = [[50.0, 60.0]]


def _group(request):
    """``register_group`` of one adapter on prepared cases, in this (fresh) process; the
    rows are written beside the request as ``group_answer.json``."""
    from benchmarks.regbench import methods
    from benchmarks.regbench.cases import load_case

    cases = [load_case(d) for d in request["cases"]]
    work = Path(request["work"])
    rows = []
    gen = methods.load(request["method"]).register_group(
        cases, work / "work", request.get("opts") or {}
    )
    for case_id, variant, pmap, info in gen:
        mapped = None if pmap is None else np.asarray(pmap(np.array(PROBE))).tolist()
        info = {k: v for k, v in (info or {}).items() if not k.startswith("_")}
        rows.append([case_id, variant, mapped, info])
    (work / "group_answer.json").write_text(
        json.dumps({"rows": rows, "calls": CALLS}, default=str)
    )
    return 0


def run_group(method, cases, work, opts=None, **fake):
    """``register_group`` of ``method`` (``valis_mirage``: the REAL ``bin/register.py``;
    ``valis``) on ``cases`` with this stand-in installed, in a fresh process (the
    pipeline's module rebinds ``multiprocessing.cpu_count``). Returns the generator's
    rows as ``(case_id, variant, map, info)``: ``map`` is the row's point map evaluated
    at ``PROBE`` (None for a failed row), ``info`` its JSON-safe record."""
    import subprocess

    root = Path(__file__).resolve().parents[3]
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pinned = (
        {"version": "1.0.0", "rounding": "trunc"} if method == "valis_mirage" else {}
    )
    request = {
        "mode": "group",
        "method": method,
        "work": str(work),
        "cases": [str(c.dir) for c in cases],
        "opts": {"workers": 2, **(opts or {})},
        "fake": {**pinned, **fake},
        "env": {"REGBENCH_MEM_GB": "64", "REGBENCH_ATTEMPT": "1"},
    }
    (work / "group_request.json").write_text(json.dumps(request))
    environ = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    environ["PYTHONPATH"] = os.pathsep.join(
        [str(root), environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    done = subprocess.run(
        [sys.executable, __file__, str(work / "group_request.json")],
        cwd=root,
        env=environ,
        capture_output=True,
        text=True,
        timeout=900,
    )
    answer = work / "group_answer.json"
    if not answer.exists():
        raise AssertionError(done.stdout[-3000:] + done.stderr[-3000:])
    return [tuple(r) for r in json.loads(answer.read_text())["rows"]]


if __name__ == "__main__":
    raise SystemExit(main())
