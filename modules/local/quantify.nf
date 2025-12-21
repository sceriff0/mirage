nextflow.enable.dsl = 2

process QUANTIFY {
    tag "${channel_tiff.simpleName}"
    label 'gpu'
    container "${params.container.quantification}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/quantification/by_marker", mode: 'copy'

    input:
    tuple path(channel_tiff), path(seg_mask)

    output:
    path "${channel_tiff.simpleName}_quant.csv", emit: individual_csv

    script:
    """
    # Run quantification on this single channel TIFF
    quantify.py \\
        --channel_tiff ${channel_tiff} \\
        --mask_file ${seg_mask} \\
        --outdir . \\
        --output_file ${channel_tiff.simpleName}_quant.csv \\
        --min_area ${params.quant_min_area}
    """
}

process MERGE_QUANT_CSVS {
    tag "merge_quantification"
    label 'process_low'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/quantification", mode: 'copy'

    input:
    path individual_csvs

    output:
    path "merged_quant.csv", emit: merged_csv

    script:
    """
    #!/usr/bin/env python3
    import pandas as pd
    from pathlib import Path
    import sys

    # Load all individual CSVs
    csv_files = sorted(Path('.').glob('*_quant.csv'))

    if not csv_files:
        print("ERROR: No quantification CSVs found", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(csv_files)} quantification CSVs...")

    # Identify which CSV is from the reference (contains DAPI)
    reference_csv = None
    other_csvs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'DAPI' in df.columns:
            reference_csv = (csv_file, df)
            print(f"  - {csv_file.name}: REFERENCE with DAPI")
        else:
            other_csvs.append((csv_file, df))
            print(f"  - {csv_file.name}: {len(df.columns)-8} markers")  # -8 for morphology columns

    if reference_csv is None:
        print("ERROR: No reference CSV with DAPI column found", file=sys.stderr)
        sys.exit(1)

    # Start with reference dataframe (has all morphological features + DAPI)
    merged = reference_csv[1].copy()
    print(f"\\nStarting with reference: {len(merged)} cells, {len(merged.columns)} columns")

    # Morphological and metadata columns to exclude from merging (already in reference)
    morphology_cols = ['label', 'y', 'x', 'area', 'eccentricity', 'perimeter',
                      'convex_area', 'axis_major_length', 'axis_minor_length']

    # Merge marker columns from other CSVs
    for csv_file, df in other_csvs:
        # Get only marker columns (exclude morphology and DAPI if present)
        marker_cols = [col for col in df.columns if col not in morphology_cols and col != 'DAPI']

        if marker_cols:
            # Select label + marker columns
            merge_df = df[['label'] + marker_cols]

            # Merge on label
            merged = merged.merge(merge_df, on='label', how='outer')
            print(f"  + Added {len(marker_cols)} markers from {csv_file.name}")

    # Reorder columns: morphology first, then DAPI, then other markers
    morpho_present = [col for col in morphology_cols if col in merged.columns]
    marker_cols_all = [col for col in merged.columns if col not in morphology_cols and col != 'DAPI']

    final_column_order = morpho_present + ['DAPI'] + sorted(marker_cols_all)
    merged = merged[final_column_order]

    # Save merged CSV
    merged.to_csv('merged_quant.csv', index=False)
    print(f"\\n✓ Merged CSV saved: {len(merged)} cells, {len(merged.columns)} columns")
    print(f"  Final columns: {', '.join(merged.columns)}")
    """
}
