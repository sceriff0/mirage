"""Close VALIS 1.0.0-1.2.0's pyramid-level -1 defect without modifying the VALIS image.

VALIS-free on purpose: this runs (and is tested) where ``valis`` is not installed.
``bin/register.py`` hands it ``slide_io.BioFormatsSlideReader`` to patch.

THE DEFECT THIS CLOSES -- reproduced 2026-09-08 with VALIS's own geometry code
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

Slides larger than the requested size fail too, through the mask term: the needed source
dimension is roughly ``max_non_rigid_dim / (tissue-mask extent)``, so it depends on image
CONTENT and no size check can predict it. Measured 2026-09-22: a TMA patient at 1024 px
(``-profile tma``) passed the old size preflight on every slide and still died here, the
micro-rigid ROIs showing a tissue box a small part of the frame.

THE FIX is upstream's one line, ``max(closest_img_levels[0] - 1, 0)``, applied at the
reader instead (``clamp_negative_levels``): a negative level reaches Bio-Formats as 0.
That is safe by construction -- ``setResolution()`` rejects every negative level, so the
only calls the clamp changes are calls that would have crashed -- and it is the same
value upstream's fix would pick. VALIS's next line, ``resize_img(vips_level_img,
src_img_shape_rc)``, upsamples level 0 to the size it asked for, so the non-rigid stage
runs, on no more detail than full resolution holds. The level also reaches the image
processor (``processor_cls(..., level=closest_img_level)``), but that processor is handed
the already-read image and only reads through the reader when ``image is None``
(``preprocessing.py:115``), so the reader is the one place the level matters.

Only ``BioFormatsSlideReader`` is patched: it is the reader that crashes. Other readers
index ``slide_dimensions[-1]`` in Python and silently read the smallest level instead;
changing that would change runs that currently complete, which is a separate decision.

``slides_too_small_for_non_rigid`` stays as a NOTICE: those slides now register, but
their non-rigid stage runs on upsampled full-resolution pixels, which an operator
choosing ``--max-non-rigid-dim`` should know.
"""

from __future__ import annotations

import functools
import inspect
import os
import types
from typing import Dict, Iterable, List, Tuple

import tifffile

# The BioFormatsSlideReader methods that take a pyramid level as their first parameter
# (valis_lib/slide_io.py:909, :973). get_channel() reaches Bio-Formats through
# slide2image(), so these two cover every route to setResolution().
LEVEL_METHODS = ("slide2vips", "slide2image")


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


def upsample_notice(offenders: List[Tuple[str, int]], non_rigid_dim: int) -> str:
    """What happens to slides no larger than the non-rigid size, for the task log."""
    listing = "\n".join(
        f"    {os.path.basename(name)}: {dim} px" for name, dim in offenders
    )
    return (
        f"The non-rigid registration size ({non_rigid_dim} px) is not smaller than the full "
        f"resolution of {len(offenders)} input slide(s):\n{listing}\n"
        "VALIS will read these at full resolution (pyramid level 0) and upsample them to the "
        "size it needs, so their non-rigid stage runs on no more detail than full resolution "
        "holds. A --max-non-rigid-dim below the smallest slide costs less and loses nothing "
        "for them. (VALIS 1.0.0-1.2.0 would have asked Bio-Formats for level -1 here and "
        "crashed; clamp_negative_levels in bin/utils/valis_preflight.py prevents that.)"
    )


def _clamped(fn):
    """Wrap a reader method so a negative ``level`` reaches it as 0."""

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if args:
            if args[0] is not None and args[0] < 0:
                args = (0,) + args[1:]
        elif kwargs.get("level") is not None and kwargs["level"] < 0:
            kwargs["level"] = 0
        return fn(self, *args, **kwargs)

    wrapped._mirage_level_clamp = True
    return wrapped


def clamp_negative_levels(reader_cls) -> List[str]:
    """Patch ``reader_cls`` so no negative pyramid level reaches Bio-Formats.

    Returns the names of the methods patched by THIS call: empty when they were already
    patched, and empty when a method is missing or its first parameter is not ``level``
    (a VALIS whose signatures moved) -- the caller logs that, because it means the
    defect is open again.
    """
    patched = []
    for name in LEVEL_METHODS:
        fn = getattr(reader_cls, name, None)
        if not isinstance(fn, types.FunctionType) or getattr(
            fn, "_mirage_level_clamp", False
        ):
            continue
        params = list(inspect.signature(fn).parameters)
        if len(params) < 2 or params[1] != "level":
            continue
        setattr(reader_cls, name, _clamped(fn))
        patched.append(name)
    return patched
