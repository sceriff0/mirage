"""Work around VALIS 1.0.0-1.2.0's pyramid-level -1 defect without changing the VALIS image.

Imports no ``valis``, so it is testable without it. ``bin/register.py`` hands it
``slide_io.BioFormatsSlideReader`` to patch.

``Valis.prep_images_for_large_non_rigid_registration`` picks the pyramid level per slide::

    closest_img_levels = np.where(np.max(slide_dimensions_wh, axis=1) < np.max(src_img_shape_rc))[0]
    closest_img_level = closest_img_levels[0] - 1        # registration.py:3480-3482 (1.0.0)
                                                          # same at main (1.2.0):3952-3954

When even level 0 is smaller than the source size the non-rigid stage needs, that is
``-1``. Bio-Formats ``setResolution(-1)`` then throws in every tile thread,
``get_tiles_parallel`` swallows it, and ``Valis.register()`` ends with a ``UserWarning``
and ``(None, None, None)`` instead of an error (see tests/test_valis_preflight.py).

It is not predictable from slide size alone: VALIS clamps the non-rigid size to the
smallest slide and then scales it up by the reference frame or tissue-mask extent, so
the smallest slide always comes out short, and larger slides can too depending on the
tissue mask.

The fix is upstream's ``max(closest_img_levels[0] - 1, 0)``, applied at the reader
(``clamp_negative_levels``): a negative level reaches Bio-Formats as 0. Only calls that
would have crashed change. VALIS then upsamples level 0 to the requested size. The image
processor also receives the level but reads through the reader only when ``image is
None`` (``preprocessing.py:115``), so the reader is the one place it matters.

Only ``BioFormatsSlideReader`` is patched; other readers index ``slide_dimensions[-1]``
and silently read the smallest level, and changing that would change runs that complete.

``slides_too_small_for_non_rigid`` is reported as a notice: those slides register, but
their non-rigid stage runs on upsampled full-resolution pixels.
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
