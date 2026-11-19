#!/usr/bin/env python3
"""Pre-flight scale scan: resolve `--pixel_size` for every input slide up front.

A wrong pixel size silently corrupts every area, distance and density downstream, so
this reads each image's OME header before heavy work starts. It fails on an
unresolvable `auto` and warns when a supplied number disagrees with, or cannot be
confirmed by, the file.

Reads OME metadata only, never pixel data, so it stays fast over every input.
Problems are collected across all images, so one run names every offending slide.

Exit codes
----------
0   every slide's scale was resolved (auto) or accepted (a supplied number), possibly
    with WARNING(s) logged for a disagreement or an unconfirmable slide.
1   `--pixel-size auto` was requested and at least one slide has no usable OME
    PhysicalSizeX/Y. The error names every such slide.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from logger import configure_logging, get_logger  # noqa: E402
from pixel_size import (  # noqa: E402
    AUTO,
    PIXEL_SIZE_RTOL,
    read_ome_pixel_size,
    warn_on_pixel_size_mismatch,
)

logger = get_logger(__name__)

__all__ = ["main"]


def _parse_pixel_size(raw: str) -> Optional[float]:
    """Return the configured µm/px as a float, or None for `'auto'`.

    Raises ValueError for anything that is neither.
    """
    text = raw.strip()
    if text.lower() == AUTO:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"--pixel-size {raw!r} is neither a positive number nor '{AUTO}'."
        ) from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"--pixel-size must be a positive number of micrometres per pixel, got {value}."
        )
    return value


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Resolve --pixel_size for every input slide from OME metadata only, "
        "before any heavy work is staged."
    )
    ap.add_argument(
        "--images", nargs="+", required=True, help="Input image files to scan."
    )
    ap.add_argument(
        "--pixel-size",
        required=True,
        help=f"'{AUTO}' to read each image's own OME PhysicalSizeX, or a positive number of "
        "micrometres per pixel.",
    )
    ap.add_argument(
        "--output", required=True, help="Path to write the per-slide report JSON."
    )
    return ap.parse_args(argv)


def _warn_on_heterogeneous_scales(report: dict, logger) -> None:
    """Warn (never fail) when this run's slides resolved to distinguishable scales.

    Mixed-magnification cohorts are legitimate, so this only names each distinct value
    and its slides. Values within `PIXEL_SIZE_RTOL` count as one scale.
    """
    values = sorted({info["pixel_size"] for info in report.values()})
    if len(values) < 2:
        return

    clusters: List[List[float]] = []
    for value in values:
        if clusters and abs(value - clusters[-1][-1]) <= PIXEL_SIZE_RTOL * abs(
            clusters[-1][-1]
        ):
            clusters[-1].append(value)
        else:
            clusters.append([value])
    if len(clusters) < 2:
        return

    groups = []
    for cluster in clusters:
        representative = cluster[0]
        slides = sorted(
            Path(path).name
            for path, info in report.items()
            if abs(info["pixel_size"] - representative)
            <= PIXEL_SIZE_RTOL * abs(representative)
        )
        groups.append(f"{representative:g} µm/px: {', '.join(slides)}")

    logger.warning(
        "  [SCALE HETEROGENEITY] this run's slides resolved to %d distinct pixel sizes, "
        "but are registered together and merged into one pyramid -- %s. This is expected "
        "for a legitimate mixed-magnification cohort and is not an error; each slide "
        "keeps its own resolved scale.",
        len(clusters),
        "; ".join(groups),
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: resolve and validate every image's pixel size before the run starts.

    Returns
    -------
    int
        0 when every image resolved (or was confirmed against the operator's
        value); 1 when ``--pixel-size`` could not be parsed or an image's scale
        is unresolvable.
    """
    configure_logging()
    args = _parse_args(argv)

    try:
        configured = _parse_pixel_size(args.pixel_size)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    is_auto = configured is None

    report: dict = {}
    unresolvable: List[str] = []
    unconfirmed: List[str] = []

    # Collect a verdict for every image before deciding; do not break out early.
    for image in args.images:
        path = Path(image)
        det_x, det_y = read_ome_pixel_size(path)
        detected = det_x if det_x is not None else det_y

        if is_auto:
            if detected is None:
                unresolvable.append(path.name)
                continue
            report[str(path)] = {"pixel_size": detected, "source": "metadata"}
            logger.info(
                "  %s: resolved --pixel_size %s to %g µm/px from OME metadata.",
                path.name,
                AUTO,
                detected,
            )
        else:
            mismatched = warn_on_pixel_size_mismatch(
                (det_x, det_y),
                configured,
                source=path.name,
                logger=logger,
            )
            if not mismatched and detected is None:
                unconfirmed.append(path.name)
            report[str(path)] = {"pixel_size": configured, "source": "operator"}

    _warn_on_heterogeneous_scales(report, logger)

    if is_auto and unresolvable:
        logger.error(
            "--pixel_size %s was requested but %d slide(s) carry no usable OME "
            "PhysicalSizeX/Y (absent header, absent attribute, non-positive value, or an "
            "unrecognised unit) -- their scale cannot be resolved before the run starts. "
            "Pass an explicit --pixel_size instead of '%s' for: %s.",
            AUTO,
            len(unresolvable),
            AUTO,
            ", ".join(sorted(unresolvable)),
        )
        return 1

    if unconfirmed:
        # Not an error: most WSI formats carry no OME scale; the configured value wins.
        logger.warning(
            "  [SCALE UNCONFIRMED] --pixel_size %g was supplied, but it could not be "
            "verified against OME metadata for %d slide(s) (no PhysicalSizeX/Y present): "
            "%s. This is normal for many WSI formats and is not an error -- proceeding "
            "with the configured %g µm/px for these slides.",
            configured,
            len(unconfirmed),
            ", ".join(sorted(unconfirmed)),
            configured,
        )

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Pre-flight scale scan: resolved %d slide(s); wrote %s",
        len(report),
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
