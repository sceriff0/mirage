"""The one pixel-size rule.

``params.pixel_size`` owns every micrometre conversion in this pipeline: the µm
measurements written into ``cells.geojson``, the ``PhysicalSize`` stamped on the
published pyramid, and InstantSeg's rescaling target all read it and nothing else.
That single ownership is deliberate -- it is the same "exactly one owner" rule the
resource labels and the nuclear-marker resolution follow -- but it has a failure mode.
An image whose own OME header claims a different scale is converted, registered,
segmented and quantified at the *configured* scale with nothing said about the
disagreement, and the symptom surfaces a long way from the cause: the sibling
``qupath-extension-flowpath`` sees a scale it cannot reconcile and warns there, in
QuPath, about a decision this pipeline made silently several steps earlier.

This module is the one place the two values are put side by side.
``read_ome_pixel_size`` reports what the file claims; ``warn_on_pixel_size_mismatch``
compares it against what was configured and says so, loudly, naming both numbers.

Neither function changes a number. The configured value stays authoritative, so a run's
outputs do not depend on whatever metadata happens to be embedded in its inputs -- which
in whole-slide imaging is frequently absent, wrong, or in units the writer guessed at.
The point is that the operator learns about it at ``CONVERT_IMAGE`` and can set
``--pixel_size`` correctly, rather than discovering it in QuPath after the fact.

Import convention: flat (``from pixel_size import ...``) by scripts that
``sys.path.insert(0, .../bin/utils)`` first, matching ``measurements.py`` and
``image_utils.py``.
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

# Relative tolerance for calling two pixel sizes "the same". 1% is well below any real
# objective/scanner difference (the smallest step between adjacent magnifications is
# tens of percent) and well above float-formatting noise in an OME header, where a
# 0.325 µm scale is routinely serialised as 0.32499998807907104.
PIXEL_SIZE_RTOL = 0.01

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


def unit_to_um(unit: Optional[str]) -> Optional[float]:
    """Multiplier taking a length in ``unit`` to micrometres, or None if the
    unit is one this module does not recognise.

    Exported rather than kept private because the conversion table must not
    fork: a second copy in another script would drift, and the drift would be a
    silent scale error in whichever copy fell behind. What "unrecognised" MEANS
    is deliberately left to the caller -- ``read_ome_pixel_size`` below reports
    None and lets the caller carry on, because its result is only ever used to
    WARN while ``params.pixel_size`` stays authoritative; a caller deriving an
    authoritative scale would have to raise on the same None instead.

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
    """Read ``PhysicalSizeX``/``PhysicalSizeY`` from a TIFF's OME header, in µm.

    Returns ``(None, None)`` for a file with no OME header, no ``Pixels`` element, no
    ``PhysicalSize`` attributes, or a unit this module does not recognise. Every one of
    those is ordinary rather than exceptional -- most intermediates in this pipeline
    carry no scale at all -- so none of them raises.

    Only OME metadata is read. The TIFF ``XResolution``/``ResolutionUnit`` tags are a
    different and much less reliable convention; ``bin/convert_image.py`` handles those
    for the vendor formats where they are the only thing available.
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


def warn_on_pixel_size_mismatch(
    detected: Tuple[Optional[float], Optional[float]] | Optional[float],
    configured: float,
    *,
    source: str,
    logger: logging.Logger,
    rtol: float = PIXEL_SIZE_RTOL,
) -> bool:
    """Compare an image's own scale against the configured one; warn if they disagree.

    Returns True when a mismatch was reported, so a caller can count them. The
    configured value is authoritative either way -- this function never returns a scale
    to use, precisely so that no caller can accidentally start preferring the metadata.

    An image with no embedded scale is not a mismatch. It is the normal case for the
    pipeline's own intermediates and says nothing about whether ``--pixel_size`` is
    right, so warning about it would train operators to ignore the warning that matters.
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

    A distinct type so a caller can tell "this image has no scale" apart from "this
    number is malformed", and so the message reaches the operator intact rather than as
    a bare traceback.
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

    THREE CASES, AND ONLY THREE. There is deliberately no fallback constant here: a
    repo-wide default pixel size is one scanner's value wearing everyone else's label,
    and applying it silently is how a run produces µm measurements that are uniformly
    wrong by the ratio of two objectives. Every path below either returns a number that
    came from the operator or from the image, or raises.

    * a positive number -- returned as given. The operator is asserting the scale and
      it wins over anything the file claims; ``warn_on_pixel_size_mismatch`` is what
      tells them when the file disagrees.
    * ``"auto"`` -- read ``PhysicalSizeX`` from ``image_path``'s OME header. This is
      safe at every stage of the pipeline because ``CONVERT_IMAGE`` stamps the scale it
      resolved onto its output and each rewriting step (``apply_basic_profiles``,
      ``tiled_stitch``) re-stamps it, so an image reaching any consumer carries one --
      with one exception worth naming. ``bin/register.py`` itself writes no
      ``PhysicalSize``; the registered output is believed to be re-stamped, but one
      layer down, inside VALIS itself (``Slide.warp_and_save_slide`` ->
      ``slide_io.save_ome_tiff``), which is believed to copy the *source* slide's
      physical size -- scaled to the output pyramid level -- into the new OME-XML,
      provided the source carried one. That belief was traced from VALIS's vendored
      source, not observed at runtime -- no test in this repo exercises the
      registered-output-carries-scale path end to end, so treat it as unconfirmed. VALIS
      is pinned (``modules/local/register.nf`` builds ``cdgatenbee/valis-wsi:1.0.0``),
      so this is not a silent-drift hazard -- a version bump is a deliberate act, and is
      the moment to re-verify this behaviour still holds. If it ever stopped holding, or
      the source image had no physical size to begin with (OME unit literally
      ``"px"``), the registered output would carry no scale, and a re-entry over it via
      ``--start segmentation`` / ``--start postprocessing`` would hard-fail at
      ``PREFLIGHT_SCALE`` under ``auto`` -- loudly, which is correct, but is worth an
      operator knowing the cause is upstream of this pipeline's own code. Absent or
      uninterpretable metadata RAISES -- that is the "otherwise error" half of the
      contract, and it is the whole point: ``auto`` promises to read the real scale, so
      it must fail loudly rather than substitute a guess.
    * ``None`` -- raises. ``params.pixel_size`` defaults to ``'auto'``, not null, so
      reaching this function with ``None`` means something explicitly overrode the
      default back to null/empty; the launch-time guard in
      ``ParamUtils.validatePixelSize`` rejects exactly that before any process runs, so
      getting here means that guard was bypassed.
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
