#!/usr/bin/env python3
"""Generate registration quality control images.

This script creates RGB composite images for visual assessment of registration
quality by overlaying reference and registered DAPI channels.

Examples
--------
Generate QC for single image pair:

    $ python generate_registration_qc.py \\
        --reference patient1_DAPI_CD4_CD8.ome.tif \\
        --registered patient1_DAPI_CD4_CD8_registered.ome.tif \\
        --output qc/

Batch mode with multiple registered images:

    $ python generate_registration_qc.py \\
        --reference patient1_DAPI_CD4_CD8.ome.tif \\
        --registered patient1_*_registered.ome.tif \\
        --output qc/

Custom scale factor:

    $ python generate_registration_qc.py \\
        --reference ref.tif \\
        --registered reg.tif \\
        --output qc/ \\
        --scale-factor 0.5

Output Format
-------------
For each registered image, creates:
- {basename}_QC_RGB_fullres.tif: Full resolution, compressed, ImageJ-compatible
- {basename}_QC_RGB.tif: Downsampled TIFF for ImageJ viewing
- {basename}_QC_RGB.png: Downsampled PNG for quick preview

Channel assignment in composites:
- Red: Registered image
- Green: Reference image
- Blue: 0 (black)

Perfect alignment appears yellow (red + green).
Misalignment shows red/green fringing.

Notes
-----
This is a refactored version that uses shared library modules following
DRY, KISS, and software engineering best practices.

The previous implementation duplicated code from register_cpu.py and
register_gpu.py. This version imports from lib.qc instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Import from shared library modules (eliminates code duplication)
from logger import get_logger, configure_logging
from qc import create_registration_qc

__all__ = ["main"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - reference: Path to reference image
        - registered: List of paths to registered images
        - output: Path to output directory
        - scale_factor: Downsampling factor
        - no_fullres: Skip full-resolution output
        - no_png: Skip PNG output
        - no_tiff: Skip downsampled TIFF output
        - verbose: Enable debug logging
    """
    parser = argparse.ArgumentParser(
        description="Generate registration QC comparing registered vs reference images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate QC for single image pair
  %(prog)s \\
    --reference patient1_DAPI_CD4_CD8.ome.tif \\
    --registered patient1_DAPI_CD4_CD8_registered.ome.tif \\
    --output qc/

  # Batch mode with multiple registered images
  %(prog)s \\
    --reference patient1_DAPI_CD4_CD8.ome.tif \\
    --registered patient1_*_registered.ome.tif \\
    --output qc/

Output:
  For each registered image, creates:
    - {basename}_QC_RGB_fullres.tif  (full resolution, compressed)
    - {basename}_QC_RGB.tif          (downsampled for ImageJ)
    - {basename}_QC_RGB.png          (downsampled for quick view)
        """
    )

    parser.add_argument(
        '--reference',
        type=Path,
        required=True,
        help='Path to reference image (OME-TIFF)'
    )

    parser.add_argument(
        '--registered',
        type=Path,
        nargs='+',
        required=True,
        help='Path(s) to registered image(s) (OME-TIFF). Can specify multiple files.'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for QC files'
    )

    parser.add_argument(
        '--scale-factor',
        type=float,
        default=0.25,
        help='Downsampling factor for PNG/TIFF output (default: 0.25 = 4x smaller)'
    )

    parser.add_argument(
        '--no-fullres',
        action='store_true',
        help='Skip full-resolution TIFF output (saves disk space)'
    )

    parser.add_argument(
        '--no-png',
        action='store_true',
        help='Skip PNG output'
    )

    parser.add_argument(
        '--no-tiff',
        action='store_true',
        help='Skip downsampled TIFF output'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    return parser.parse_args()


def process_single_image(
    reference_path: Path,
    registered_path: Path,
    output_dir: Path,
    scale_factor: float,
    save_fullres: bool,
    save_png: bool,
    save_tiff: bool,
    logger
) -> Tuple[bool, str]:
    """Process a single registered image to generate QC outputs.

    Parameters
    ----------
    reference_path : Path
        Path to reference image.
    registered_path : Path
        Path to registered image.
    output_dir : Path
        Output directory.
    scale_factor : float
        Downsampling factor.
    save_fullres : bool
        Whether to save full-resolution output.
    save_png : bool
        Whether to save PNG output.
    save_tiff : bool
        Whether to save downsampled TIFF output.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    success : bool
        Whether processing succeeded.
    message : str
        Success or error message.
    """
    try:
        # Generate base output path from registered image name
        base_name = registered_path.stem.replace('.ome', '')
        output_path = output_dir / f"{base_name}_QC_RGB.tif"

        # Call library function (eliminates code duplication)
        create_registration_qc(
            reference_path=reference_path,
            registered_path=registered_path,
            output_path=output_path,
            scale_factor=scale_factor,
            save_fullres=save_fullres,
            save_png=save_png,
            save_tiff=save_tiff
        )

        return True, "Success"

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to create QC for {registered_path.name}: {error_msg}")
        logger.debug("Full traceback:", exc_info=True)
        return False, error_msg


def main() -> int:
    """Main entry point for batch QC generation.

    Returns
    -------
    int
        Exit code (0 for success, 1 if any images failed).

    Notes
    -----
    This function:
    1. Configures logging using singleton pattern from lib.logger
    2. Parses and validates command-line arguments
    3. Processes each registered image in batch mode
    4. Provides summary statistics
    5. Returns appropriate exit code

    The logging is configured once at startup and shared across all modules
    that import from lib.logger, following the singleton pattern.
    """
    # Parse arguments
    args = parse_args()

    # Configure logging using singleton pattern
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(
        level=log_level,
        format_string='[%(asctime)s] %(levelname)s - %(message)s'
    )

    # Get logger for this module
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("REGISTRATION QC GENERATION")
    logger.info("=" * 80)

    # Validate reference image exists
    if not args.reference.exists():
        logger.error(f"Reference image not found: {args.reference}")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reference image: {args.reference}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Scale factor: {args.scale_factor}")
    logger.info(f"Save full-res: {not args.no_fullres}")
    logger.info(f"Save PNG: {not args.no_png}")
    logger.info(f"Save TIFF: {not args.no_tiff}")
    logger.info(f"Images to process: {len(args.registered)}")
    logger.info("")

    # Process each registered image
    success_count = 0
    failed_images: List[Tuple[Path, str]] = []

    for i, reg_path in enumerate(args.registered, 1):
        logger.info(f"Processing {i}/{len(args.registered)}: {reg_path.name}")

        # Validate file exists
        if not reg_path.exists():
            logger.warning(f"  ✗ File not found: {reg_path}")
            failed_images.append((reg_path, "File not found"))
            continue

        # Skip if it's the reference image itself
        if reg_path.resolve() == args.reference.resolve():
            logger.info(f"  ⊘ Skipping (is reference image)")
            continue

        # Process image
        success, message = process_single_image(
            reference_path=args.reference,
            registered_path=reg_path,
            output_dir=args.output,
            scale_factor=args.scale_factor,
            save_fullres=not args.no_fullres,
            save_png=not args.no_png,
            save_tiff=not args.no_tiff,
            logger=logger
        )

        if success:
            success_count += 1
            logger.info(f"  ✓ Success")
        else:
            failed_images.append((reg_path, message))
            logger.warning(f"  ✗ Failed: {message}")

        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("QC GENERATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successfully processed: {success_count}/{len(args.registered)}")

    if failed_images:
        logger.warning(f"Failed images ({len(failed_images)}):")
        for path, error in failed_images:
            logger.warning(f"  - {path.name}: {error}")
        logger.info("=" * 80)
        return 1
    else:
        logger.info("✓ All QC outputs generated successfully")
        logger.info("=" * 80)
        return 0


if __name__ == '__main__':
    sys.exit(main())
