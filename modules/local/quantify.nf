nextflow.enable.dsl = 2

process QUANTIFY {
    tag "${meta.patient_id} - ${channel_tiff.simpleName}"
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:quantification_gpu' :
        'docker://bolt3x/attend_image_analysis:quantification_gpu' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/quantification/by_marker", mode: 'copy'

    input:
    tuple val(meta), path(channel_tiff), path(seg_mask)

    output:
    tuple val(meta), path("${channel_tiff.simpleName}_quant.csv"), emit: individual_csv
    path "versions.yml"                                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    echo "Sample: ${meta.patient_id}"

    # Run quantification on this single channel TIFF
    quantify.py \\
        --channel_tiff ${channel_tiff} \\
        --mask_file ${seg_mask} \\
        --outdir . \\
        --output_file ${channel_tiff.simpleName}_quant.csv \\
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
    touch ${channel_tiff.simpleName}_quant.csv

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

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:quantification_gpu' :
        'docker://bolt3x/attend_image_analysis:quantification_gpu' }"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/quantification", mode: 'copy'

    input:
    tuple val(meta), path(individual_csvs)

    output:
    tuple val(meta), path("merged_quant.csv"), emit: merged_csv
    path "versions.yml"                       , emit: versions

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

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        pandas: stub
    END_VERSIONS
    """
}
