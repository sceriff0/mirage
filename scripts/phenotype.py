#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import json
import tifffile as tiff
from typing import List, Union
import argparse
from scipy.stats import norm, zscore
import logging
from datetime import datetime

from scripts._common import ensure_dir, setup_file_logger

def setup_logging(output_dir, log_file=None):
    """Simple logging setup: ensure output dir and optional file logger."""
    ensure_dir(output_dir)
    if log_file:
        setup_file_logger(log_file)

def labels_to_phenotype(arr, phenotype_df):
    """Map label array to phenotype numbers."""
    logging.info("Mapping labels to phenotypes")
    try:
        map_arr = phenotype_df[['label', 'phenotype_num']].to_numpy()
        max_val = max(map_arr[:, 0].max(), arr.max()) + 1
        lookup = np.zeros(max_val + 1, dtype=map_arr[:, 1].dtype)
        lookup[map_arr[:, 0]] = map_arr[:, 1]
        remapped_arr = lookup[arr]
        logging.info("Label to phenotype mapping completed")
        return remapped_arr
    except Exception as e:
        logging.error(f"Error in labels_to_phenotype: {str(e)}")
        raise

def run_phenotyping_pipeline(cell_df, mask, output_dir):
    """
    Run the original pipeline (from the first script)
    This simulates the original functions with same logic
    """
    # Import the original functions (assuming they're available)
    # For this example, we'll simulate the key steps
    
    # Step 2: Reorder columns (original column order)
    cols_order = [
        'y', 'x', 'eccentricity', 'perimeter', 'convex_area', 'area',
        'axis_major_length', 'axis_minor_length', 'label',
        'ARID1A', 'CD14', 'CD163', 'CD3', 'CD4', 'CD45', 'CD8', 'FOXP3',
        'L1CAM', 'P53', 'PANCK', 'PAX2', 'PD1', 'PDL1', 'SMA', 'GZMB', 'CD74', 'VIMENTIN', 'DAPI'
    ]
    
    cell_df = cell_df[cols_order]
    
    # Step 3: Quality filtering (original logic)
    nuc_thres = np.percentile(cell_df['DAPI'], 1.0)
    size_thres = np.percentile(cell_df['area'], 1.0)
    cell_df_filtered = cell_df[(cell_df['DAPI'] > nuc_thres) & (cell_df['area'] > size_thres)]
    
    # Step 4: Normalization (original format function logic)
    list_out = ['eccentricity', 'perimeter', 'convex_area', 'axis_major_length', 'axis_minor_length']
    list_keep = ['DAPI', 'x', 'y', 'area', 'label']
    
    # Remove excluded columns
    dfin = cell_df_filtered.drop(list_out, axis=1)
    df_loc = dfin.loc[:, list_keep]
    dfz = dfin.drop(list_keep, axis=1)
    
    # Apply z-score normalization (original method)
    from scipy.stats import zscore
    dfz1 = pd.DataFrame(zscore(dfz, 0), index=dfz.index, columns=dfz.columns)
    dfz_all = pd.concat([dfz1, df_loc], axis=1, join="inner")
    
    # Step 5: Noise removal (original logic)
    last_marker = 'VIMENTIN'
    col_num_last_marker = dfz_all.columns.get_loc(last_marker)
    
    # Calculate thresholds
    dfz_copy = dfz_all.copy()
    dfz_copy["Count"] = dfz_all.iloc[:, :col_num_last_marker + 1].ge(0).sum(axis=1)
    dfz_copy["z_sum"] = dfz_all.iloc[:, :col_num_last_marker + 1].sum(axis=1)
    
    count_threshold = dfz_copy["Count"].quantile(1 - 0.01)
    z_sum_threshold = dfz_copy["z_sum"].quantile(1 - 0.01)
    
    # Remove noise
    df_nn = dfz_copy#[~((dfz_copy.iloc[:, :col_num_last_marker + 1].ge(0).sum(axis=1) > count_threshold) | 
                     # (dfz_copy.iloc[:, :col_num_last_marker + 1].sum(axis=1) > z_sum_threshold))].copy().reset_index(drop=True)

    # Step 6: Phenotyping (original logic)
    mark_list = ['CD163', 'CD14', 'CD45', 'CD3', 'CD8', 'CD4', 'FOXP3', 'PANCK', 'VIMENTIN', 'SMA', 'L1CAM', 'PAX2', 'CD74', 'GZMB', 'PD1', 'PDL1']
    cutoff_list = [0.7, 0.9, 0.4, 0.2, 0.4, 0.9, 1.3, 0.2, 0.2, 0.2, 0.3, 1, 1.3, 1.2, 1.5, 0.4]

    df_nn['pheno_markers'] = [[] for _ in range(len(df_nn))]

    for m, c in zip(mark_list, cutoff_list): 
    
        sel = df_nn[df_nn[m] >= c]
        sel_idx = sel.index
        
        for idx in sel_idx:
            df_nn.at[idx, 'pheno_markers'].append(m)

    
    # Immune compartment
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x), 'phenotype'] = "Immune"
    
    # cd4+
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" in x and "CD4" in x and "FOXP3" in x and "CD8" not in x), 'phenotype'] = "CD4 T regulatory"
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" in x and "CD4" in x and "FOXP3" not in x and "CD8" not in x), 'phenotype'] = "T helper"
    
    # cd8+
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" in x and "GZMB" not in x and "FOXP3" not in x and "CD8" in x and "CD4" not in x), 'phenotype'] = "T cytotoxic"
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" in x and "GZMB" not in x and "FOXP3" in x and "CD8" in x and "CD4" not in x), 'phenotype'] = "CD8 T regulatory"
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" in x and "GZMB"  in x and "CD8" in x and "CD4" not in x), 'phenotype'] = "activated T cytotoxic"
    
    # macrophages
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" not in x and "CD14" in x and "CD163" in x), 'phenotype'] = "Macrophages"
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" not in x and "CD14" in x and "CD163" not in x), 'phenotype'] = "M1"
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" in x and "CD3" not in x and "CD14" not in x and "CD163" in x), 'phenotype'] = "M2"
    
    # Stroma
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" not in x), 'phenotype'] = "Stroma"
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" not in x and "SMA" in x), 'phenotype'] = "Stroma"
    
    # Tumor
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" not in x and "SMA" not in x and "PANCK" in x), 'phenotype'] = "PANCK+ Tumor"
    df_nn.loc[df_nn['pheno_markers'].apply(lambda x: "CD45" not in x and "SMA" not in x and "VIMENTIN" in x), 'phenotype'] = "VIM+ Tumor"

    # Add numeric labels
    pheno_complete = df_nn['phenotype'].value_counts().index.values
    for pp, p in enumerate(pheno_complete):
        sel = df_nn[df_nn['phenotype'] == p].index
        df_nn.loc[sel, 'phenotype_num'] = pp + 1
    
    df_nn['phenotype_num'] = df_nn['phenotype_num'].astype(int)
    
    # Create phenotype mask
    phenotype_mask = labels_to_phenotype(mask, df_nn)
    #label_to_phenotype = dict(zip(df_nn['label'], df_nn['phenotype_num']))
    #max_label = max(mask.max(), max(label_to_phenotype.keys()))
    #lookup = np.zeros(max_label + 1, dtype=np.int32)
   # 
    #for label, phenotype in label_to_phenotype.items():
    #    if label <= max_label:
    #        lookup[label] = phenotype
    
    #phenotype_mask = lookup[mask]
    
    return df_nn, phenotype_mask


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell phenotyping analysis based on marker expression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--cell_data',
        required=True,
        help='Path to input CSV file containing cell data'
    )

    parser.add_argument(
        '--segmentation_mask',
        required=True,
        help='Path to input segmentation mask file (npy)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help='Path to output directory where results will be saved'
    )

    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse arguments
        args = parse_arguments()
        cell_df = pd.read_csv(args.cell_data).drop_duplicates(subset='label', keep='first')
        mask = np.load(args.segmentation_mask)
        # Setup logging
        setup_logging(args.output_dir)
        logging.info("Starting cell phenotyping pipeline")
        logging.info(f"Arguments: {vars(args)}")
        phenotypes_data, phenotypes_mask = run_phenotyping_pipeline(
            cell_df.copy(), mask.copy(), args.output_dir
        )
        
        logging.info(f"Saving {os.path.join(args.output_dir, 'phenotypes_data.csv')} and {os.path.join(args.output_dir, 'phenotypes_mask.tiff')}")
        phenotypes_data.to_csv(os.path.join(args.output_dir, 'phenotypes_data.csv')) 
        tiff.imwrite(os.path.join(args.output_dir, 'phenotypes_mask.tiff'), phenotypes_mask)
        logging.info("Cell phenotyping pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise
