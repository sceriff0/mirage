"""Refuse, before the JVM starts, input that VALIS 1.0.0-1.2.0 cannot register.

VALIS-free on purpose: this runs (and is tested) where ``valis`` is not installed.

THE DEFECT THIS GUARDS AGAINST -- reproduced 2026-09-08 with VALIS's own geometry code
on the dimensions from a failed TMA run (see tests/test_valis_preflight.py's docstring
for the full chain). ``Valis.prep_images_for_large_non_rigid_registration`` chooses the
pyramid level to read for each slide as::

    closest_img_levels = np.where(np.max(slide_dimensions_wh, axis=1) < np.max(src_img_shape_rc))[0]
    closest_img_level = closest_img_levels[0] - 1        # registration.py:3480-3482 (1.0.0)
                                                          # same at main (1.2.0):3952-3954

When even level 0 -- full resolution -- is smaller than the source dimension the
non-rigid stage needs, that is ``-1``. ``slide2vips(-1)`` sizes the tile grid from
``slide_dimensions[-1]`` (Python's negative index: the SMALLEST level, silently) and
every tile thread calls Bio-Formats ``setResolution(-1)``, which throws
``IllegalArgumentException`` (``loci.formats.FormatReader``: ``no < 0 ||
no >= getResolutionCount()``). ``get_tiles_parallel`` swallows it (``print(e); pass``),
leaves ``tile`` unbound, and ``UnboundLocalError: local variable 'tile' referenced
before assignment`` reaches ``Valis.register()``'s catch-all, which prints it as a
``UserWarning``, kills the JVM and returns ``(None, None, None)``. Nothing raises.

VALIS's own clamp does not prevent it. When some slide is smaller than
``max_non_rigid_registration_dim_px`` it lowers that value to the smallest slide's
largest dimension -- and then computes the source dimension it needs as
``processed_max * s`` where ``s`` scales the REFERENCE's processed frame (or, with
``create_masks=True``, the tissue-mask bounding box, which is smaller) up to the
clamped value. The smallest slide therefore always comes out short: by one pixel from
the ceil with no mask (2721 vs 2720), by a lot with one (4096 vs 2720 for a mask
covering 60 % of the frame). The clamp firing is a deterministic predictor of the
crash, and it fires exactly when some slide's level-0 largest dimension is below the
requested size. ``<=`` also covers the equal case, which the ceil makes fail.

Slides larger than the requested size can still fail through the mask term, which
cannot be predicted from sizes alone; bin/register.py catches that case after the fact
by treating ``register()``'s ``None`` return as the failure it is. The fix that closes
both is one line upstream -- ``max(closest_img_levels[0] - 1, 0)``, after which level 0
is read and ``resize_img`` (already the next call) upsamples it -- and is not applied
here: the pipeline runs the unmodified ``cdgatenbee/valis-wsi`` image.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import tifffile


def level0_max_dims(paths: Iterable[str]) -> Dict[str, int]:
    """Largest spatial (Y or X) extent of the full-resolution level of each slide.

    Reads only the TIFF directory. The channel axis of a CYX stack and the sample
    axis of an interleaved RGB page are ignored, so a 40-plex stack reports its
    height/width, never 40.
    """
    out: Dict[str, int] = {}
    for path in paths:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = series.axes
            spatial = [n for ax, n in zip(axes, series.shape) if ax in ("Y", "X")]
            if len(spatial) != 2:
                # No axis labels worth trusting: fall back to the first page's own
                # (height, width), which is what VALIS's slide_dimensions_wh[0] is.
                spatial = list(tif.pages[0].shape[:2])
            out[path] = int(max(spatial))
    return out


def slides_too_small_for_non_rigid(
    level0_max: Dict[str, int], non_rigid_dim: int
) -> List[Tuple[str, int]]:
    """The slides whose full resolution is no larger than the non-rigid size.

    Each is a slide VALIS will try to read at pyramid level -1. Smallest first, so
    the message leads with the one that sets the ceiling.
    """
    return sorted(
        ((name, dim) for name, dim in level0_max.items() if dim <= non_rigid_dim),
        key=lambda nd: (nd[1], nd[0]),
    )


def refusal_message(offenders: List[Tuple[str, int]], non_rigid_dim: int) -> str:
    """The operator-facing reason and the two remedies, in one string."""
    smallest = offenders[0][1]
    listing = "\n".join(
        f"    {os.path.basename(name)}: {dim} px" for name, dim in offenders
    )
    return (
        "Refusing to start VALIS: the non-rigid registration size "
        f"({non_rigid_dim} px) is not smaller than the full resolution of "
        f"{len(offenders)} input slide(s):\n{listing}\n"
        "VALIS 1.0.0-1.2.0 selects pyramid level -1 for such a slide "
        "(registration.py, prep_images_for_large_non_rigid_registration), Bio-Formats "
        "rejects setResolution(-1), the reader swallows the error, and Valis.register() "
        "kills the JVM and returns None -- after the whole rigid stage has run. "
        "Its own size clamp does not prevent this; it only changes the number by which "
        "the slide falls short.\n"
        "Remedies:\n"
        f"  1. Register at a non-rigid size below the smallest slide: --max-non-rigid-dim "
        f"{smallest - 1} is the hard ceiling (pipeline: memory_mode = 'custom' with "
        f"reg_valis_max_non_rigid_dim = {smallest - 1}). Leave a margin: the source size "
        "VALIS actually needs is (processed size) x (non-rigid size / tissue-mask extent), "
        "so a slide only slightly larger than the setting still fails when the mask covers "
        f"part of the frame. {smallest // 2} is safe whenever tissue covers at least half "
        "of the reference frame.\n"
        "  2. Use the tiled backend, which has no such limit: registration_method = 'tiled'."
    )
