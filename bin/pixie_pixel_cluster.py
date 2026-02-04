#!/usr/bin/env python3
"""
Pixie Pixel Clustering - Automated version of notebook workflow.

Based on: https://github.com/angelolab/pixie/blob/main/templates/1_Pixie_Cluster_Pixels.ipynb

This script performs pixel-level clustering using Self-Organizing Maps (SOM)
followed by consensus clustering to identify pixel phenotypes.

Steps:
1. Create pixel matrix from multi-channel images
2. Train pixel SOM on subset of data
3. Assign all pixels to SOM clusters
4. Perform consensus clustering for meta-clusters
5. Generate cluster average profiles
6. Export parameters for cell clustering

Author: ATEIA Pipeline (adapted from ark-analysis)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
    args = parser.parse_args()

    # Import ark modules (delayed import to check availability)
    try:
        from alpineer import io_utils
        from ark.phenotyping import (
            pixel_meta_clustering,
            pixel_som_clustering,
            pixie_preprocessing
        )
    except ImportError as e:
        print(f"ERROR: Failed to import ark-analysis modules: {e}", file=sys.stderr)
        print("Please ensure ark-analysis==0.6.4 is installed.", file=sys.stderr)
        sys.exit(1)

    # Setup directories
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    fovs = [args.fov_name]
    channels = args.channels

    print(f"Pixie Pixel Clustering")
    print(f"=" * 50)
    print(f"FOV: {args.fov_name}")
    print(f"Channels: {', '.join(channels)}")
    print(f"Blur factor: {args.blur_factor}")
    print(f"Subset proportion: {args.subset_proportion}")
    print(f"Max K: {args.max_k}")
    print(f"Cap: {args.cap}")
    print(f"Seed: {args.seed}")
    print()

    # Output directory structure (matching notebook)
    pixel_output_dir = "pixel_output"
    pixel_data_dir = os.path.join(pixel_output_dir, 'pixel_mat_data')
    pixel_subset_dir = os.path.join(pixel_output_dir, 'pixel_mat_subset')
    norm_vals_name = os.path.join(pixel_output_dir, 'channel_norm_post_rowsum.feather')

    os.makedirs(os.path.join(base_dir, pixel_data_dir), exist_ok=True)
    os.makedirs(os.path.join(base_dir, pixel_subset_dir), exist_ok=True)

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
        multiprocess=False,
        batch_size=1
    )
    print("  Pixel matrix created successfully.")

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
        multiprocess=False,
        batch_size=1
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
        multiprocess=False,
        batch_size=1
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

    # =========================================================================
    # Step 5: Export parameters for cell clustering
    # =========================================================================
    print("Step 5: Exporting parameters for cell clustering...")
    cell_clustering_params = {
        'fovs': fovs,
        'channels': channels,
        'segmentation_dir': args.seg_dir,
        'seg_suffix': args.seg_suffix,
        'pixel_data_dir': pixel_data_dir,
        'pc_chan_avg_som_cluster_name': pc_chan_avg_som_cluster_name,
        'pc_chan_avg_meta_cluster_name': pc_chan_avg_meta_cluster_name
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

    print()
    print("Pixel clustering complete!")
    print(f"Results saved to: {base_dir}/{pixel_output_dir}")


if __name__ == '__main__':
    main()
