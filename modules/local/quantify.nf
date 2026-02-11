nextflow.enable.dsl = 2

/*
 * QUANTIFY - Marker intensity quantification
 *
 * Measures per-cell marker intensities from single-channel TIFFs using
 * segmentation masks. Computes morphological features and intensity statistics.
 *
 * Input: Single-channel TIFF and segmentation mask
 * Output: Per-channel quantification CSV with cell measurements
 */
process QUANTIFY {
    tag "${meta.patient_id} - ${channel_tiff.simpleName}"
    label 'process_medium'

    container 'docker://bolt3x/attend_image_analysis:quantification_gpu'

    input:
    tuple val(meta), path(channel_tiff), path(seg_mask)

    output:
    tuple val(meta), path("${meta.id}_quant.csv"), emit: individual_csv
    path "versions.yml"                           , emit: versions
    path("*.size.csv")                            , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    // Extract channel name from filename (split_multichannel.py creates files like "PANCK.tiff")
    def channel_name = channel_tiff.simpleName
    """
    # Log input sizes for tracing (sum of channel_tiff + seg_mask, -L follows symlinks)
    tiff_bytes=\$(stat -L --printf="%s" ${channel_tiff})
    mask_bytes=\$(stat -L --printf="%s" ${seg_mask})
    total_bytes=\$((tiff_bytes + mask_bytes))
    echo "${task.process},${meta.id},${channel_tiff.name}+${seg_mask.name},\${total_bytes}" > ${meta.id}.QUANTIFY.size.csv

    echo "Sample: ${meta.patient_id}"
    echo "Channel: ${channel_name}"

    # Run quantification on this single channel TIFF
    quantify.py \\
        --channel_tiff ${channel_tiff} \\
        --channel-name ${channel_name} \\
        --mask_file ${seg_mask} \\
        --outdir . \\
        --output_file ${meta.id}_quant.csv \\
        --min_area ${params.quant_min_area} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "unknown")
        scikit-image: \$(python -c "import skimage; print(skimage.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch ${meta.id}_quant.csv
    echo "STUB,${meta.id},stub,0" > ${meta.id}.QUANTIFY.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        pandas: stub
        scikit-image: stub
    END_VERSIONS
    """
}

process MERGE_QUANT_CSVS {
    tag "${meta.patient_id}"
    label 'process_low'

    container 'docker://bolt3x/attend_image_analysis:quantification_gpu'

    input:
    tuple val(meta), path(individual_csvs)

    output:
    tuple val(meta), path("merged_quant.csv"), emit: merged_csv
    path "versions.yml"                       , emit: versions
    path("*.size.csv")                        , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    #!/usr/bin/env python3
    import pandas as pd
    from pathlib import Path
    import sys
    import os

    # Log input size for tracing (sum of all CSV files)
    csv_files = sorted(Path('.').glob('*_quant.csv'))
    total_bytes = sum(f.stat().st_size for f in csv_files)
    with open('${meta.patient_id}.MERGE_QUANT_CSVS.size.csv', 'w') as f:
        f.write(f"${task.process},${meta.patient_id},csvs/,{total_bytes}\\n")

    print("Sample: ${meta.patient_id}")

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
    # Use left join to preserve all cells from the reference
    # Fill missing values with 0 (cell not detected in this channel)
    print("\\nValidating CSV compatibility...")
    reference_cells = set(reference_csv[1]['label'])

    for csv_file, df in other_csvs:
        # Validate cell labels match
        other_cells = set(df['label'])
        missing = reference_cells - other_cells
        extra = other_cells - reference_cells

        if missing:
            print(f"  WARNING: {csv_file.name}: Missing {len(missing)} cells from reference")
            print(f"     These cells will have 0 intensity for this channel")
        if extra:
            print(f"  WARNING: {csv_file.name}: Has {len(extra)} extra cells not in reference")
            print(f"     Extra cells will be ignored (not in segmentation mask)")

        # Get only marker columns (exclude morphology and DAPI if present)
        marker_cols = [col for col in df.columns if col not in morphology_cols and col != 'DAPI']

        if marker_cols:
            # Select label + marker columns
            merge_df = df[['label'] + marker_cols]

            # Cells missing from this channel will have NaN, which we fill with 0
            merged = merged.merge(merge_df, on='label', how='left')

            # Fill NaN with 0 (cell not detected in this channel = no signal)
            for col in marker_cols:
                merged[col] = merged[col].fillna(0.0)

            print(f"  + Added {len(marker_cols)} markers from {csv_file.name}")

    # Validate no cells were lost (should never happen with left join)
    cells_lost = len(reference_csv[1]) - len(merged)
    if cells_lost > 0:
        print(f"\\nCRITICAL ERROR: Lost {cells_lost} cells during merge")
        print(f"  Reference had {len(reference_csv[1])} cells, merged has {len(merged)}")
        print(f"  This should not happen with left join - investigation needed")
        sys.exit(1)
    else:
        print(f"\\nAll {len(merged)} cells from reference preserved")

    # Reorder columns: morphology first, then DAPI, then other markers
    morpho_present = [col for col in morphology_cols if col in merged.columns]
    marker_cols_all = [col for col in merged.columns if col not in morphology_cols and col != 'DAPI']

    final_column_order = morpho_present + ['DAPI'] + sorted(marker_cols_all)
    merged = merged[final_column_order]

    # Save merged CSV
    merged.to_csv('merged_quant.csv', index=False)
    print(f"\\nMerged CSV saved: {len(merged)} cells, {len(merged.columns)} columns")
    print(f"  Final columns: {', '.join(merged.columns)}")

    # Write versions file
    with open('versions.yml', 'w') as f:
        f.write('"${task.process}":\\n')
        f.write(f'    python: {sys.version.split()[0]}\\n')
        f.write(f'    pandas: {pd.__version__}\\n')
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch merged_quant.csv
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.MERGE_QUANT_CSVS.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        pandas: stub
    END_VERSIONS
    """
}
