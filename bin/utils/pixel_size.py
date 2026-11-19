"""The one pixel-size rule.

``params.pixel_size`` owns every micrometre conversion: the µm measurements in
``cells.geojson``, the ``PhysicalSize`` on the published pyramid, and InstanSeg's
rescaling target. An input whose OME header claims a different scale is still processed
at the configured scale.

``read_ome_pixel_size`` reports what the file claims; ``warn_on_pixel_size_mismatch``
warns, naming both values, when it differs from the configured one. Neither changes a
number: the configured value stays authoritative, since WSI metadata is often missing or
wrong. The warning appears at CONVERT_IMAGE, not later in QuPath.

Imported flat (``from pixel_size import ...``) after ``bin/utils`` is on sys.path.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import tifffile

__all__ = [
    "AUTO",
    "read_ome_pixel_size",
    "resolve_pixel_size",
    "warn_on_pixel_size_mismatch",
    "unit_to_um",
    "PIXEL_SIZE_RTOL",
]

# The literal `--pixel_size` value meaning "take the scale from the image itself".
AUTO = "auto"

# Relative tolerance for "the same" pixel size: far below any real magnification step,
# far above float noise (0.325 is often serialised as 0.32499998807907104).
PIXEL_SIZE_RTOL = 0.01

# An unrecognised unit yields None rather than a mis-scaled comparison.
_UNIT_TO_UM = {
    "µm": 1.0,
    "um": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "mm": 1e3,
    "millimeter": 1e3,
    "cm": 1e4,
    "m": 1e6,
}


def unit_to_um(unit: Optional[str]) -> Optional[float]:
    """Multiplier taking a length in ``unit`` to micrometres, or None if unrecognised.

    Public so no script keeps a second conversion table. ``None`` (attribute absent) maps
    to 1.0: OME's 2016-06 schema defaults to micrometres. What an unrecognised unit means
    is the caller's decision.
    """
    return _UNIT_TO_UM.get((unit or "µm").strip())


def _to_um(raw: Optional[str], unit: Optional[str]) -> Optional[float]:
    """Convert one OME PhysicalSize attribute to µm, or None if it cannot be trusted."""
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    # OME's default unit when the attribute is absent is µm, per the 2016-06 schema.
    factor = unit_to_um(unit)
    if factor is None:
        return None
    return value * factor


def read_ome_pixel_size(
    path: str | Path,
) -> Tuple[Optional[float], Optional[float]]:
    """Read ``PhysicalSizeX``/``PhysicalSizeY`` from a TIFF's OME header, in µm.

    Returns ``(None, None)`` when there is no OME header, no ``Pixels`` element, no
    ``PhysicalSize`` or an unrecognised unit; most intermediates carry no scale, so none
    of these raises.

    Only OME metadata is read; TIFF resolution tags are left to ``bin/convert_image.py``.
    """
    try:
        with tifffile.TiffFile(str(path)) as tif:
            ome_xml = getattr(tif, "ome_metadata", None)
            if not ome_xml or not isinstance(ome_xml, str):
                return (None, None)
            root = ET.fromstring(ome_xml)
    except Exception:
        # An unreadable header means no scale: (None, None). Never fail a run on metadata.
        return (None, None)

    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return (None, None)

    return (
        _to_um(pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit")),
        _to_um(pixels.get("PhysicalSizeY"), pixels.get("PhysicalSizeYUnit")),
    )


def warn_on_pixel_size_mismatch(
    detected: Tuple[Optional[float], Optional[float]] | Optional[float],
    configured: float,
    *,
    source: str,
    logger: logging.Logger,
    rtol: float = PIXEL_SIZE_RTOL,
) -> bool:
    """Compare an image's own scale against the configured one; warn if they disagree.

    Returns True when a mismatch was reported. Never returns a scale, so no caller can
    start preferring the metadata.

    An image with no embedded scale is not a mismatch (it is normal for intermediates),
    so it is not warned about.
    """
    if not isinstance(detected, tuple):
        detected = (detected, detected)
    det_x, det_y = detected

    reported = False
    for axis, det in (("X", det_x), ("Y", det_y)):
        if det is None:
            continue
        if abs(det - configured) <= rtol * abs(configured):
            continue
        logger.warning(
            "  [SCALE MISMATCH] %s reports PhysicalSize%s = %g µm/px, but the run is "
            "configured with --pixel_size %g (%.0f%% apart). Every µm measurement -- "
            "GeoJSON centroids and areas, the published pyramid's PhysicalSize, "
            "InstantSeg's rescaling -- uses the configured %g. If the image is right, "
            "re-run with --pixel_size %g.",
            source,
            axis,
            det,
            configured,
            100.0 * abs(det - configured) / abs(configured),
            configured,
            det,
        )
        reported = True

    if not reported and det_x is None and det_y is None:
        logger.debug(
            "  %s carries no OME pixel size; using the configured %g µm/px.",
            source,
            configured,
        )
    return reported


class PixelSizeError(ValueError):
    """`--pixel_size auto` was asked for and the image carries no usable scale.

    A distinct type, so "no scale" can be told apart from a malformed number.
    """


def resolve_pixel_size(
    configured: str | float | None,
    image_path: str | Path | None = None,
    *,
    detected: Tuple[Optional[float], Optional[float]] | Optional[float] = None,
    source: str | None = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Turn the configured ``--pixel_size`` into the µm/px this run will actually use.

    There is no fallback constant: every path returns a value from the operator or the
    image, or raises.

    * a positive number: returned as given; it wins over the file
      (``warn_on_pixel_size_mismatch`` reports disagreement).
    * ``"auto"``: read ``PhysicalSizeX`` from ``image_path``'s OME header; absent or
      unreadable metadata raises. CONVERT_IMAGE stamps the resolved scale and the
      rewriting steps (``apply_basic_profiles``, ``tiled_stitch``) re-stamp it.
      ``bin/register.py`` writes none itself and relies on VALIS copying the source
      slide's physical size into the registered OME-XML; this is untested end to end, so
      a ``--start segmentation`` re-entry under ``auto`` can fail at PREFLIGHT_SCALE if
      the registered file has no scale.
    * ``None``: raises. The default is ``'auto'`` and ``ParamUtils.validatePixelSize``
      rejects null at launch, so reaching this means that guard was bypassed.
    """
    label = source or (str(image_path) if image_path is not None else "<no image>")

    if configured is None or (isinstance(configured, str) and not configured.strip()):
        raise PixelSizeError(
            "--pixel_size is unset. Pass a positive number of micrometres per pixel, "
            f"or '{AUTO}' to read it from the image's own OME metadata."
        )

    if isinstance(configured, str) and configured.strip().lower() == AUTO:
        # `detected` lets a caller that already has the scale pass it in, e.g.
        # convert_image.py for vendor formats that read_ome_pixel_size cannot read.
        if detected is None:
            if image_path is None:
                raise PixelSizeError(
                    f"--pixel_size {AUTO} needs an image to read the scale from, "
                    "but this step was given none."
                )
            detected = read_ome_pixel_size(image_path)
        if not isinstance(detected, tuple):
            detected = (detected, detected)
        det_x, det_y = detected
        detected = det_x if det_x is not None else det_y
        if detected is None:
            raise PixelSizeError(
                f"--pixel_size {AUTO} was requested but {label} carries no usable OME "
                "PhysicalSizeX/Y (absent header, absent attribute, non-positive value, "
                "or an unrecognised unit). Pass an explicit --pixel_size instead of "
                f"'{AUTO}' for this input."
            )
        if logger is not None:
            logger.info(
                "  %s: resolved --pixel_size %s to %g µm/px from OME metadata.",
                label,
                AUTO,
                detected,
            )
        return float(detected)

    try:
        value = float(configured)
    except (TypeError, ValueError):
        raise PixelSizeError(
            f"--pixel_size {configured!r} is neither a number nor '{AUTO}'."
        ) from None
    if value <= 0:
        raise PixelSizeError(
            f"--pixel_size must be a positive number of micrometres per pixel, got {value}."
        )
    return value
