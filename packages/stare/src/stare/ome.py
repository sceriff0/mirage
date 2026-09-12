"""The OME-TIFF header the stitch writes, and the pixel size it is stamped with.

Exactly the functions ``stare.stages.stitch`` needs, copied from the mirage
pipeline's ``bin/utils/ome_io.py`` (``ome_metadata``, ``ome_tiff_writer``) and
``bin/utils/pixel_size.py`` (``resolve_pixel_size`` and the readers behind it),
so the package imports nothing from the pipeline. The copies are kept identical
to the originals by ``tests/test_stare_package_copies_do_not_drift.py`` on the
mirage side -- it compares each definition's AST (docstrings aside) and writes
one array through both writers -- so edit both or neither.

Two rules the originals carry, restated because they are the point:

* ``None`` OMITS RATHER THAN SUBSTITUTES in the header. A missing scale means
  "the file did not say"; inventing one is the failure ``resolve_pixel_size``
  exists to prevent. Same for channel names: anonymous is honest, mislabelled
  is not.
* KEY ORDER IS PART OF THE OUTPUT. tifffile walks the metadata dict to build
  the XML, so the order in ``ome_metadata`` is the attribute order in the header
  of every registered slide.
"""

from __future__ import annotations

import contextlib
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import tifffile

__all__ = [
    "AUTO",
    "PixelSizeError",
    "ome_metadata",
    "ome_tiff_writer",
    "read_ome_pixel_size",
    "resolve_pixel_size",
    "unit_to_um",
]

_MICRON = "µm"

# The literal `--pixel-size` value meaning "take the scale from the image itself".
AUTO = "auto"

# OME-XML records the unit alongside the value, and writers do use more than µm.
# A value read without its unit is not a pixel size, it is a number -- so an
# unrecognised unit yields None rather than a silently mis-scaled comparison.
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


def ome_metadata(
    channels: Optional[Sequence[str]],
    pixel_size_um: Union[float, Tuple[Optional[float], Optional[float]], None],
    *,
    axes: str = "CYX",
    pixel_size_z_um: Optional[float] = None,
) -> dict:
    """The OME metadata dict tifffile serialises into the registered slide's header.

    ``pixel_size_um`` may be one number (X and Y equal) or an ``(x, y)`` pair.
    ``None`` anywhere omits that key rather than inventing a value.
    """
    if isinstance(pixel_size_um, tuple):
        px_x, px_y = pixel_size_um
    else:
        px_x = px_y = pixel_size_um

    md: dict = {"axes": axes}
    if channels:
        md["Channel"] = {"Name": list(channels)}
    if px_x is not None:
        md["PhysicalSizeX"] = px_x
        md["PhysicalSizeXUnit"] = _MICRON
    if px_y is not None:
        md["PhysicalSizeY"] = px_y
        md["PhysicalSizeYUnit"] = _MICRON
    if pixel_size_z_um is not None:
        md["PhysicalSizeZ"] = pixel_size_z_um
        md["PhysicalSizeZUnit"] = _MICRON
    return md


@contextlib.contextmanager
def ome_tiff_writer(path, *, bigtiff: bool = True, ome: bool = True):
    """An open ``tifffile.TiffWriter`` for a caller that streams tiles into one file.

    ``bigtiff`` defaults on: a registered slide is written uncompressed at full
    resolution, so classic TIFF's 32-bit offsets overflow the moment the output
    crosses 4 GB. ``ome=True`` is explicit because ``ome=False`` SUPPRESSES the
    header outright rather than deferring to the ``.ome.tif`` suffix.
    """
    writer = tifffile.TiffWriter(str(path), bigtiff=bigtiff, ome=ome)
    try:
        yield writer
    finally:
        writer.close()


def unit_to_um(unit: Optional[str]) -> Optional[float]:
    """Multiplier taking a length in ``unit`` to micrometres, or None if unrecognised.

    ``None`` means the attribute was absent, which OME's 2016-06 schema defines
    as micrometres -- so it maps to 1.0, not to "unrecognised".
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
    """``PhysicalSizeX``/``PhysicalSizeY`` from a TIFF's OME header, in µm.

    ``(None, None)`` for a file with no OME header, no ``Pixels`` element, no
    ``PhysicalSize`` attributes, or a unit this module does not recognise -- all
    ordinary rather than exceptional, so none of them raises.
    """
    try:
        with tifffile.TiffFile(str(path)) as tif:
            ome_xml = getattr(tif, "ome_metadata", None)
            if not ome_xml or not isinstance(ome_xml, str):
                return (None, None)
            root = ET.fromstring(ome_xml)
    except Exception:
        # An unreadable or malformed header means "no scale available", which is
        # exactly the (None, None) case. Reading metadata must never fail a run.
        return (None, None)

    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return (None, None)

    return (
        _to_um(pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit")),
        _to_um(pixels.get("PhysicalSizeY"), pixels.get("PhysicalSizeYUnit")),
    )


class PixelSizeError(ValueError):
    """``--pixel-size auto`` was asked for and the image carries no usable scale."""


def resolve_pixel_size(
    configured: str | float | None,
    image_path: str | Path | None = None,
    *,
    detected: Tuple[Optional[float], Optional[float]] | Optional[float] = None,
    source: str | None = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Turn the configured ``--pixel-size`` into the µm/px the stitch will stamp.

    Three cases and only three: a positive number is returned as given;
    ``"auto"`` reads ``PhysicalSizeX`` from ``image_path``'s OME header (or the
    ``detected`` pair a caller already read) and RAISES when there is none;
    ``None`` raises. There is deliberately no fallback constant.
    """
    label = source or (str(image_path) if image_path is not None else "<no image>")

    if configured is None or (isinstance(configured, str) and not configured.strip()):
        raise PixelSizeError(
            "--pixel_size is unset. Pass a positive number of micrometres per pixel, "
            f"or '{AUTO}' to read it from the image's own OME metadata."
        )

    if isinstance(configured, str) and configured.strip().lower() == AUTO:
        # `detected` lets a caller that has ALREADY read the scale hand it over rather
        # than have this function re-open the file. convert_image.py is the reason: its
        # input is a vendor format (.czi, .ndpi, .svs) that read_ome_pixel_size's
        # tifffile/OME-XML path cannot speak, but aicsimageio has already given it
        # physical_pixel_sizes. One rule, two ways to feed it -- not two rules.
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
