#!/usr/bin/env python3
"""
Pixie Pixel Clustering - Automated version of notebook workflow.

Based on: https://github.com/angelolab/pixie/blob/main/templates/1_Pixie_Cluster_Pixels.ipynb

This script performs pixel-level clustering using Self-Organizing Maps (SOM)
followed by consensus clustering to identify pixel phenotypes.

Steps:
1. Create pixel matrix from multi-channel images (with optional FOV tiling)
2. Train pixel SOM on subset of data
3. Assign all pixels to SOM clusters
4. Perform consensus clustering for meta-clusters
5. Generate cluster average profiles
6. Export parameters for cell clustering

Supports FOV tiling for large images and Pixie's internal multiprocessing
for parallel processing of multiple FOV tiles.

Author: ATEIA Pipeline (adapted from ark-analysis)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import gc

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.tiling import (
    create_fov_directory_structure,
    needs_tiling,
    save_tile_positions,
    calculate_tile_grid,
)


# =============================================================================
# Performance optimization: Monkey-patch SOM cluster mapping with larger batch
# Original pixie uses batch_size=100, causing 21M iterations for 2B pixels.
# This patch uses batch_size=10000 for ~100x fewer iterations.
# =============================================================================
def _patched_generate_som_clusters(self, external_data: pd.DataFrame) -> np.ndarray:
    """Optimized SOM cluster mapping with larger batch size."""
    from pyFlowSOM import map_data_to_nodes
    from alpineer.misc_utils import verify_in_list

    weights_cols = self.weights.columns.values
    verify_in_list(weights_cols=weights_cols, external_data_cols=external_data.columns.values)

    # Pre-cast types once (not per-batch)
    weights_data = self.weights.values.astype(np.float64)
    external_values = external_data[weights_cols].values.astype(np.float64)

    cluster_labels = []
    batch_size = 10000  # Was 100 in original!

    for i in range(0, external_values.shape[0], batch_size):
        cluster_labels.append(map_data_to_nodes(
            weights_data,
            external_values[i:i + batch_size]
        )[0])

    if not cluster_labels:
        return np.empty(0)
    return np.concatenate(cluster_labels)


def _apply_som_optimization():
    """Apply the monkey-patch to PixieSOMCluster."""
    from ark.phenotyping.cluster_helpers import PixieSOMCluster
    PixieSOMCluster.generate_som_clusters = _patched_generate_som_clusters


def get_image_dimensions(tiff_dir: str, fov_name: str) -> tuple:
    """Get dimensions of the first channel image in FOV directory."""
    fov_dir = Path(tiff_dir) / fov_name
    if not fov_dir.is_dir():
        raise ValueError(f"FOV directory not found: {fov_dir}")

    tiff_files = list(fov_dir.glob("*.tif")) + list(fov_dir.glob("*.tiff"))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in: {fov_dir}")

    with tifffile.TiffFile(tiff_files[0]) as tif:
        shape = tif.pages[0].shape
    return shape[:2]  # (height, width)


def align_norm_vals_channel_order(base_dir: str, norm_vals_name: str, channels: list) -> None:
    """Ensure channel normalization columns follow the requested channel order."""
    import feather

    norm_vals_path = os.path.join(base_dir, norm_vals_name)
    if not os.path.exists(norm_vals_path):
        raise FileNotFoundError(f"Normalization values file not found: {norm_vals_path}")

    norm_vals = feather.read_dataframe(norm_vals_path)
    norm_cols = list(norm_vals.columns)
    expected_cols = list(channels)

    if set(norm_cols) != set(expected_cols):
        missing = [ch for ch in expected_cols if ch not in norm_cols]
        extra = [ch for ch in norm_cols if ch not in expected_cols]
        raise ValueError(
            f"Normalization file channels do not match requested channels. "
            f"Missing: {missing}, Extra: {extra}"
        )

    if norm_cols != expected_cols:
        norm_vals = norm_vals.loc[:, expected_cols]
        feather.write_dataframe(norm_vals, norm_vals_path, compression='uncompressed')
        print("  Reordered normalization columns to match requested channel order.")
    else:
        print("  Normalization column order already matches requested channels.")


def main():
    parser = argparse.ArgumentParser(
        description='Pixie Pixel Clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tiff_dir', required=True,
                        help='Directory containing FOV subdirectories with channel TIFFs')
    parser.add_argument('--seg_dir', required=True,
                        help='Directory containing segmentation masks')
    parser.add_argument('--seg_suffix', default='_cell_mask.tif',
                        help='Suffix for segmentation mask files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--channels', required=True, nargs='+',
                        help='List of channels for clustering')
    parser.add_argument('--fov_name', required=True,
                        help='Name of the FOV (patient_id)')
    parser.add_argument('--blur_factor', type=int, default=2,
                        help='Gaussian blur sigma for preprocessing')
    parser.add_argument('--subset_proportion', type=float, default=0.1,
                        help='Fraction of pixels for SOM training')
    parser.add_argument('--num_passes', type=int, default=1,
                        help='Number of SOM training passes')
    parser.add_argument('--max_k', type=int, default=20,
                        help='Maximum number of meta-clusters')
    parser.add_argument('--cap', type=int, default=3,
                        help='Z-score capping value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    # New tiling and multiprocessing arguments
    parser.add_argument('--tile_size', type=int, default=2048,
                        help='FOV tile size for splitting large images')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for multiprocessing (default: auto from CPU count)')
    parser.add_argument('--multiprocess', action='store_true', default=False,
                        help='Enable Pixie multiprocessing for parallel FOV processing')
    args = parser.parse_args()

    # Import ark modules (delayed import to check availability)
    try:
        from alpineer import io_utils
        from ark.phenotyping import (
            pixel_meta_clustering,
            pixel_som_clustering,
            pixie_preprocessing
        )
        # Apply SOM optimization (batch_size 100 -> 10000)
        _apply_som_optimization()
    except ImportError as e:
        print(f"ERROR: Failed to import ark-analysis modules: {e}", file=sys.stderr)
        print("Please ensure ark-analysis==0.6.4 is installed.", file=sys.stderr)
        sys.exit(1)

    # Setup directories
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    channels = args.channels

    # Determine batch_size with conservative default (original pixie uses batch_size=5 max)
    if args.batch_size is None:
        args.batch_size = 3  # Conservative default

    # Cap batch_size to prevent memory issues with spawn multiprocessing
    MAX_BATCH_SIZE = 4
    args.batch_size = min(args.batch_size, MAX_BATCH_SIZE)

    print(f"Pixie Pixel Clustering")
    print(f"=" * 50)
    print(f"FOV: {args.fov_name}")
    print(f"Channels: {', '.join(channels)}")
    print(f"Blur factor: {args.blur_factor}")
    print(f"Subset proportion: {args.subset_proportion}")
    print(f"Max K: {args.max_k}")
    print(f"Cap: {args.cap}")
    print(f"Seed: {args.seed}")
    print(f"Tile size: {args.tile_size}")
    print(f"Multiprocess: {args.multiprocess}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Get image dimensions to check if tiling is needed
    height, width = get_image_dimensions(args.tiff_dir, args.fov_name)
    print(f"Image dimensions: {width}x{height}")

    do_tiling = needs_tiling(height, width, args.tile_size)
    tile_positions = None

    if do_tiling:
        n_rows, n_cols = calculate_tile_grid(height, width, args.tile_size)
        n_tiles = n_rows * n_cols
        print(f"Tiling enabled: {n_rows}x{n_cols} = {n_tiles} tiles")

        # Get channel TIFF paths
        fov_dir = Path(args.tiff_dir) / args.fov_name
        channel_tiffs = [fov_dir / f"{ch}.tif" for ch in channels]

        # Check for .tiff extension if .tif not found
        channel_tiffs = []
        for ch in channels:
            tif_path = fov_dir / f"{ch}.tif"
            tiff_path = fov_dir / f"{ch}.tiff"
            if tif_path.exists():
                channel_tiffs.append(tif_path)
            elif tiff_path.exists():
                channel_tiffs.append(tiff_path)
            else:
                print(f"ERROR: Channel file not found for {ch}", file=sys.stderr)
                sys.exit(1)

        # Get cell mask path
        cell_mask = Path(args.seg_dir) / f"{args.fov_name}{args.seg_suffix}"
        if not cell_mask.exists():
            print(f"ERROR: Cell mask not found: {cell_mask}", file=sys.stderr)
            sys.exit(1)

        # Create tiled FOV directory structure
        tiled_dir = Path(base_dir) / "tiled_fovs"
        tiff_dir, seg_dir, fovs, tile_positions = create_fov_directory_structure(
            channel_tiffs=channel_tiffs,
            cell_mask=cell_mask,
            output_dir=tiled_dir,
            tile_size=args.tile_size,
            patient_id=args.fov_name
        )

        # Update paths to use tiled structure
        args.tiff_dir = str(tiff_dir)
        args.seg_dir = str(seg_dir)

        print(f"Created {len(fovs)} FOV tiles: {fovs[:3]}{'...' if len(fovs) > 3 else ''}")
    else:
        fovs = [args.fov_name]
        print(f"No tiling needed (image fits in {args.tile_size}x{args.tile_size})")

    print()

    # Validate channels exist in TIFF directory (check first FOV)
    tiff_fov_dir = os.path.join(args.tiff_dir, fovs[0])
    if os.path.isdir(tiff_fov_dir):
        available = sorted([f.rsplit('.', 1)[0] for f in os.listdir(tiff_fov_dir)
                            if f.endswith(('.tiff', '.tif'))])
        print(f"Available channels in FOV: {available}")
        missing = [ch for ch in channels if ch not in available]
        if missing:
            print(f"ERROR: Requested channels not found: {missing}", file=sys.stderr)
            print(f"Available channels are: {available}", file=sys.stderr)
            sys.exit(1)
    print()

    # Output directory structure (matching notebook)
    pixel_output_dir = "pixel_output"
    pixel_data_dir = os.path.join(pixel_output_dir, 'pixel_mat_data')
    pixel_subset_dir = os.path.join(pixel_output_dir, 'pixel_mat_subset')
    norm_vals_name = os.path.join(pixel_output_dir, 'channel_norm_post_rowsum.feather')

    os.makedirs(os.path.join(base_dir, pixel_data_dir), exist_ok=True)
    os.makedirs(os.path.join(base_dir, pixel_subset_dir), exist_ok=True)

    # Multiprocessing settings
    use_multiprocess = args.multiprocess and len(fovs) > 1
    batch_size = args.batch_size if use_multiprocess else 1

    if use_multiprocess:
        print(f"Multiprocessing enabled: batch_size={batch_size}, FOVs={len(fovs)}")
    else:
        print(f"Sequential processing: {len(fovs)} FOV(s)")

    # Warn if tile count is very high
    if len(fovs) > 100:
        print(f"WARNING: High tile count ({len(fovs)}). Consider increasing tile_size.")
        print(f"  Current: {args.tile_size}. For 50k images, use 4096-8192.")
    print()

    # =========================================================================
    # Step 1: Create pixel matrix
    # =========================================================================
    print("Step 1: Creating pixel matrix...")
    pixie_preprocessing.create_pixel_matrix(
        fovs=fovs,
        channels=channels,
        base_dir=base_dir,
        tiff_dir=args.tiff_dir,
        seg_dir=args.seg_dir,
        img_sub_folder=None,
        seg_suffix=args.seg_suffix,
        pixel_output_dir=pixel_output_dir,
        data_dir=pixel_data_dir,
        subset_dir=pixel_subset_dir,
        norm_vals_name=norm_vals_name,
        blur_factor=args.blur_factor,
        subset_proportion=args.subset_proportion,
        seed=args.seed,
        multiprocess=use_multiprocess,
        batch_size=batch_size
    )
    print("  Pixel matrix created successfully.")
    align_norm_vals_channel_order(base_dir, norm_vals_name, channels)
    gc.collect()  # Free memory before SOM training

    # =========================================================================
    # Step 2: Train pixel SOM
    # =========================================================================
    print("Step 2: Training pixel SOM...")
    pixel_som_weights_name = os.path.join(pixel_output_dir, 'pixel_som_weights.feather')
    pc_chan_avg_som_cluster_name = os.path.join(pixel_output_dir, 'pixel_channel_avg_som_cluster.csv')
    pc_chan_avg_meta_cluster_name = os.path.join(pixel_output_dir, 'pixel_channel_avg_meta_cluster.csv')

    pixel_pysom = pixel_som_clustering.train_pixel_som(
        fovs=fovs,
        channels=channels,
        base_dir=base_dir,
        subset_dir=pixel_subset_dir,
        norm_vals_name=norm_vals_name,
        som_weights_name=pixel_som_weights_name,
        num_passes=args.num_passes,
        seed=args.seed
    )
    print("  SOM training complete.")
    gc.collect()  # Free training data memory before cluster assignment

    # =========================================================================
    # Step 3: Assign pixels to SOM clusters
    # =========================================================================
    print("Step 3: Assigning pixels to SOM clusters...")
    pixel_som_clustering.cluster_pixels(
        fovs=fovs,
        channels=channels,
        base_dir=base_dir,
        pixel_pysom=pixel_pysom,
        data_dir=pixel_data_dir,
        multiprocess=use_multiprocess,
        batch_size=batch_size
    )

    pixel_som_clustering.generate_som_avg_files(
        fovs=fovs,
        channels=channels,
        base_dir=base_dir,
        pixel_pysom=pixel_pysom,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name
    )
    print("  SOM cluster assignment complete.")
    gc.collect()  # Free memory before consensus clustering

    # =========================================================================
    # Step 4: Consensus clustering for meta-clusters
    # =========================================================================
    print("Step 4: Running consensus clustering...")
    pixel_cc = pixel_meta_clustering.pixel_consensus_cluster(
        fovs=fovs,
        channels=channels,
        base_dir=base_dir,
        max_k=args.max_k,
        cap=args.cap,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name,
        multiprocess=use_multiprocess,
        batch_size=batch_size,
        seed=args.seed
    )

    pixel_meta_clustering.generate_meta_avg_files(
        fovs=fovs,
        channels=channels,
        base_dir=base_dir,
        pixel_cc=pixel_cc,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name,
        pc_chan_avg_meta_cluster_name=pc_chan_avg_meta_cluster_name
    )
    print("  Consensus clustering complete.")
    gc.collect()  # Free memory after consensus clustering

    # =========================================================================
    # Step 4b: Create pixel_meta_cluster_rename column (default identity mapping)
    # =========================================================================
    # In the interactive notebook workflow, users rename clusters which creates
    # the pixel_meta_cluster_rename column. For automated pipelines, we create
    # this column as an identity mapping (same values as pixel_meta_cluster).
    print("Step 4b: Creating pixel_meta_cluster_rename column...")
    import feather

    pixel_data_full_path = os.path.join(base_dir, pixel_data_dir)
    renamed_count = 0
    for fov in fovs:
        feather_path = os.path.join(pixel_data_full_path, f"{fov}.feather")
        if os.path.exists(feather_path):
            df = feather.read_dataframe(feather_path)
            if 'pixel_meta_cluster' in df.columns and 'pixel_meta_cluster_rename' not in df.columns:
                df['pixel_meta_cluster_rename'] = df['pixel_meta_cluster']
                feather.write_dataframe(df, feather_path, compression='uncompressed')
                renamed_count += 1
    print(f"  pixel_meta_cluster_rename column created for {renamed_count} FOV(s).")

    # Also add pixel_meta_cluster_rename to the channel average CSVs
    for csv_path in [pc_chan_avg_som_cluster_name, pc_chan_avg_meta_cluster_name]:
        full_csv_path = os.path.join(base_dir, csv_path) if not os.path.isabs(csv_path) else csv_path
        if os.path.exists(full_csv_path):
            csv_df = pd.read_csv(full_csv_path)
            if 'pixel_meta_cluster' in csv_df.columns and 'pixel_meta_cluster_rename' not in csv_df.columns:
                # Rename the column (not copy) since downstream expects pixel_meta_cluster_rename
                csv_df = csv_df.rename(columns={'pixel_meta_cluster': 'pixel_meta_cluster_rename'})
                csv_df.to_csv(full_csv_path, index=False)
                print(f"  Renamed pixel_meta_cluster -> pixel_meta_cluster_rename in {os.path.basename(csv_path)}")

    # =========================================================================
    # Step 5: Save tile positions (if tiling was used)
    # =========================================================================
    tile_positions_path = None
    if tile_positions is not None:
        print("Step 5a: Saving tile positions...")
        tile_positions_path = os.path.join(base_dir, pixel_output_dir, 'tile_positions.json')
        save_tile_positions(
            tile_positions=tile_positions,
            output_path=Path(tile_positions_path),
            patient_id=args.fov_name,
            tile_size=args.tile_size,
            original_height=height,
            original_width=width
        )
        print(f"  Tile positions saved to: {tile_positions_path}")

    # =========================================================================
    # Step 6: Export parameters for cell clustering
    # =========================================================================
    print("Step 5b: Exporting parameters for cell clustering...")
    cell_clustering_params = {
        'fovs': fovs,
        'original_fov': args.fov_name,
        'channels': channels,
        'segmentation_dir': args.seg_dir,
        'seg_suffix': args.seg_suffix,
        'pixel_data_dir': pixel_data_dir,
        'pc_chan_avg_som_cluster_name': pc_chan_avg_som_cluster_name,
        'pc_chan_avg_meta_cluster_name': pc_chan_avg_meta_cluster_name,
        'is_tiled': do_tiling,
        'tile_size': args.tile_size if do_tiling else None,
        'tile_positions_path': tile_positions_path,
        'original_height': height,
        'original_width': width
    }

    params_path = os.path.join(base_dir, pixel_output_dir, 'cell_clustering_params.json')
    with open(params_path, 'w') as f:
        json.dump(cell_clustering_params, f, indent=2)
    print(f"  Parameters saved to: {params_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    # Read meta cluster summary
    meta_avg_path = os.path.join(base_dir, pc_chan_avg_meta_cluster_name)
    if os.path.exists(meta_avg_path):
        meta_avg = pd.read_csv(meta_avg_path)
        n_meta_clusters = len(meta_avg)
        print()
        print(f"Summary")
        print(f"-" * 50)
        print(f"  Pixel meta-clusters identified: {n_meta_clusters}")
        print(f"  Channels used: {len(channels)}")
        print(f"  FOVs processed: {len(fovs)}")
        if do_tiling:
            print(f"  Tiling: {len(fovs)} tiles from {args.fov_name}")
            print(f"  Original dimensions: {width}x{height}")

    print()
    print("Pixel clustering complete!")
    print(f"Results saved to: {base_dir}/{pixel_output_dir}")


if __name__ == '__main__':
    main()
