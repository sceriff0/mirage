#!/usr/bin/env python3
"""Compute VALIS registrar and save as pickle.

This script performs the initial VALIS registration to compute transforms,
then saves the registrar object as a pickle file for parallel slide processing.

The registrar pickle contains all transformation matrices and alignment information
needed to warp individual slides in separate processes.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

# Disable Numba caching to avoid file locator errors
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

from valis import registration

from _common import ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    "compute_registrar",
    "save_registrar_pickle",
]


def compute_registrar(
    input_dir: str,
    reference_markers: Optional[list[str]] = None,
    max_processed_dim: int = 1800,
    max_non_rigid_dim: int = 3500,
    micro_reg_fraction: float = 0.25,
    num_features: int = 5000,
) -> registration.Valis:
    """
    Compute VALIS registrar with all transformation matrices.

    Parameters
    ----------
    input_dir : str
        Directory containing preprocessed OME-TIFF files.
    reference_markers : list of str, optional
        List of channel names to use for registration reference.
    max_processed_dim : int, optional
        Maximum dimension for processed images. Default is 1800.
    max_non_rigid_dim : int, optional
        Maximum dimension for non-rigid registration. Default is 3500.
    micro_reg_fraction : float, optional
        Fraction of image size for micro-registration. Default is 0.25.
    num_features : int, optional
        Number of features for registration. Default is 5000.

    Returns
    -------
    registrar : Valis
        Computed VALIS registrar object with all transforms.
    """
    logger.info("=" * 80)
    logger.info("COMPUTING VALIS REGISTRAR")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Reference markers: {reference_markers}")
    logger.info(f"Max processed dim: {max_processed_dim}")
    logger.info(f"Max non-rigid dim: {max_non_rigid_dim}")
    logger.info(f"Micro-registration fraction: {micro_reg_fraction}")
    logger.info(f"Number of features: {num_features}")
    logger.info("")

    start_time = time.time()

    # Initialize registrar
    logger.info("Initializing VALIS registrar...")
    registrar = registration.Valis(
        src_dir=input_dir,
        dst_dir=".",  # No output yet
        reference_img_f=reference_markers[0] if reference_markers else None,
        name="ateia_registration"
    )

    # Register slides - this computes all transforms
    logger.info("Running registration to compute transforms...")
    logger.info("This will compute:")
    logger.info("  - Rigid registration transforms")
    logger.info("  - Non-rigid registration transforms")
    logger.info("  - Micro-registration transforms")
    logger.info("")

    rigid_registrar, non_rigid_registrar, error_df = registrar.register(
        reference_channel=reference_markers if reference_markers else None,
        max_processed_image_dim_px=max_processed_dim,
        max_non_rigid_registraion_dim_px=max_non_rigid_dim,
    )

    # Micro-registration for high-resolution refinement
    logger.info("")
    logger.info("Computing micro-registration transforms...")
    try:
        _, micro_error = registrar.register_micro(
            max_non_rigid_registraion_dim_px=max_non_rigid_dim,
            micro_rigid_registrar_cls=None,  # Use default
        )
        logger.info("✓ Micro-registration complete")
    except Exception as e:
        logger.warning(f"⚠ Micro-registration failed: {e}")
        logger.warning("Continuing without micro-registration transforms...")

    elapsed = time.time() - start_time

    # Log registration summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("REGISTRATION COMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Number of slides: {len(registrar.slide_dict)}")
    logger.info(f"Reference image: {registrar.reference_img_f}")
    logger.info(f"Computation time: {elapsed:.2f}s")
    logger.info("")

    # Log slide information
    logger.info("Registered slides:")
    for slide_name in registrar.slide_dict.keys():
        logger.info(f"  - {slide_name}")
    logger.info("")

    return registrar


def save_registrar_pickle(
    registrar: registration.Valis,
    output_path: str,
) -> None:
    """
    Save registrar object as pickle file.

    Parameters
    ----------
    registrar : Valis
        Computed VALIS registrar object.
    output_path : str
        Path to save pickle file.

    Notes
    -----
    The pickle file contains:
    - All transformation matrices (rigid, non-rigid, micro)
    - Reference image information
    - Slide metadata and alignment parameters
    - Channel information
    """
    logger.info(f"Saving registrar pickle: {output_path}")

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    ensure_dir(str(output_dir))

    # Save registrar as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(registrar, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Verify pickle was saved
    pickle_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Registrar pickle saved ({pickle_size_mb:.2f} MB)")

    # Verify pickle can be loaded
    logger.info("Verifying pickle can be loaded...")
    with open(output_path, 'rb') as f:
        test_registrar = pickle.load(f)
    logger.info(f"✓ Pickle verified (contains {len(test_registrar.slide_dict)} slides)")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute VALIS registrar and save as pickle',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing preprocessed OME-TIFF files'
    )

    parser.add_argument(
        '--output-pickle',
        type=str,
        required=True,
        help='Path to save registrar pickle file'
    )

    # Optional registration parameters
    parser.add_argument(
        '--reference-markers',
        type=str,
        nargs='+',
        default=None,
        help='Channel names to use for registration reference'
    )

    parser.add_argument(
        '--max-processed-dim',
        type=int,
        default=1800,
        help='Maximum dimension for processed images'
    )

    parser.add_argument(
        '--max-non-rigid-dim',
        type=int,
        default=3500,
        help='Maximum dimension for non-rigid registration'
    )

    parser.add_argument(
        '--micro-reg-fraction',
        type=float,
        default=0.25,
        help='Fraction of image size for micro-registration'
    )

    parser.add_argument(
        '--num-features',
        type=int,
        default=5000,
        help='Number of features for registration'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    # Compute registrar
    registrar = compute_registrar(
        input_dir=args.input_dir,
        reference_markers=args.reference_markers,
        max_processed_dim=args.max_processed_dim,
        max_non_rigid_dim=args.max_non_rigid_dim,
        micro_reg_fraction=args.micro_reg_fraction,
        num_features=args.num_features,
    )

    # Save as pickle
    save_registrar_pickle(
        registrar=registrar,
        output_path=args.output_pickle,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ REGISTRAR COMPUTATION AND SAVE COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
