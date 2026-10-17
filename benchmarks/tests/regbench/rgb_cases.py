"""Small brightfield (RGB) cases for the tests of the brightfield input rules.

Each slide is the selfcheck's nuclear channel turned into a stain-like RGB image (dark
bluish nuclei on a white slide, Beer-Lambert with a haematoxylin-like absorbance), written
as a uint8 OME-TIFF of three samples per pixel (``photometric=rgb``, axes ``YXS``): the
layout of a brightfield scanner's slide, and the one VALIS's reader calls RGB. With
``interleaved=False`` it is three separate channels ``R``, ``G``, ``B`` instead
(``imageio.write_channels``), which VALIS's reader does NOT call RGB.
The case's one-channel proxy of each slide (``ref_nuclear`` / ``mov_nuclear``) is its
inverted luminance, written with ``imageio.write_nuclear``: what a dataset module writes
for STARE. Nothing here is a real brightfield slide.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import tifffile

from benchmarks.regbench import imageio
from benchmarks.regbench.cases import load_case, write_case

# a haematoxylin-like absorbance per channel (R, G, B), and BT.601 luma
ABSORBANCE = np.array([0.65, 0.70, 0.29])
LUMA = np.array([0.299, 0.587, 0.114])


def to_rgb(plane):
    """A nuclear plane (bright nuclei) as ``(3, H, W)`` uint8: dark nuclei on white."""
    a = np.asarray(plane, np.float64)
    od = 1.6 * (a - a.min()) / max(float(a.max() - a.min()), 1.0)
    rgb = 255.0 * np.exp(-od[None] * ABSORBANCE[:, None, None])
    return rgb.round().clip(0, 255).astype(np.uint8)


def grey(rgb_chw):
    """Inverted luminance of ``(3, H, W)`` uint8 as uint8."""
    lum = np.tensordot(LUMA, np.asarray(rgb_chw, np.float64), axes=1)
    return (255.0 - lum).round().clip(0, 255).astype(np.uint8)


def write_rgb(dest, rgb_chw, interleaved=True, pixel_size_um=0.5):
    """``(3, H, W)`` uint8 as an RGB OME-TIFF (module docstring)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not interleaved:
        imageio.write_channels(dest, list(rgb_chw), ["R", "G", "B"], pixel_size_um)
        return
    tifffile.imwrite(
        str(dest),
        np.ascontiguousarray(np.moveaxis(rgb_chw, 0, -1)),
        photometric="rgb",
        tile=(256, 256),
        ome=True,
        metadata={
            "axes": "YXS",
            "PhysicalSizeX": pixel_size_um,
            "PhysicalSizeY": pixel_size_um,
        },
    )


def write_rgb_slide(src, dest, channel=0, interleaved=True):
    """Channel ``channel`` of the OME-TIFF ``src`` as an RGB slide at ``dest`` and its
    grey proxy beside it. Returns ``(rgb path, proxy path)``."""
    plane = tifffile.imread(str(src))
    plane = plane[channel] if plane.ndim == 3 else plane
    rgb = to_rgb(plane)
    dest = Path(dest)
    write_rgb(dest, rgb, interleaved)
    proxy = dest.with_name(dest.name.replace(".ome.tif", ".grey.ome.tif"))
    imageio.write_nuclear(proxy, grey(rgb), 0.5)
    return str(dest), str(proxy)


def rgb_case(case, root, case_id=None, group=None, mov=None, interleaved=True):
    """``case`` (a fluorescence case) as a brightfield one: RGB slides of its nuclear
    channel and their grey proxies under ``root``. ``mov`` replaces its moving slide's
    source (a patient's rounds)."""
    root = Path(root)
    case_id = case_id or case.case_id
    ref_img, ref_proxy = write_rgb_slide(
        case.ref_image,
        root / "slides" / "reference.ome.tif",
        int(case.extra.get("ref_channel", 0)),
        interleaved,
    )
    mov_img, mov_proxy = write_rgb_slide(
        mov or case.mov_image,
        root / "slides" / f"{case_id}.ome.tif",
        int(case.extra.get("mov_channel", 0)),
        interleaved,
    )
    h, w = case.ref_hw
    extra = {
        k: v
        for k, v in case.extra.items()
        if k not in ("ref_channel", "mov_channel", "channel_names", "dots_channel")
    }
    out = dataclasses.replace(
        case,
        case_id=case_id,
        group=group or case.group,
        modality="brightfield",
        ref_image=ref_img,
        mov_image=mov_img,
        ref_nuclear=ref_proxy,
        mov_nuclear=mov_proxy,
        diagonal=math.hypot(h, w),
        extra={**extra, "settings": {"proxy": "grey", "reduce": 1}},
    )
    return load_case(write_case(root / "cases", out))


def read_rgb(path):
    """An RGB slide written here as ``(H, W, 3)``, whichever layout it has."""
    a = tifffile.imread(str(path))
    return np.moveaxis(a, 0, -1) if a.shape[0] == 3 and a.ndim == 3 else a
