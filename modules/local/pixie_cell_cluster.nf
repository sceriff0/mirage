nextflow.enable.dsl = 2

/*
 * PIXIE_CELL_CLUSTER - Cell-level clustering using Pixie
 *
 * Aggregates pixel clusters at the cell level and performs cell clustering.
 * Uses Self-Organizing Maps (SOM) followed by consensus clustering.
 *
 * Based on: https://github.com/angelolab/pixie (ark-analysis 0.6.4)
 *
 * Input:
 *   - Pixel cluster data (from PIXIE_PIXEL_CLUSTER)
 *   - Merged quantification CSV (cell table)
 *   - Cell segmentation mask
 *   - Cell clustering parameters JSON
 * Output:
 *   - Cluster counts per cell
 *   - Cell cluster assignments
 *   - Cluster profiles
 *   - Updated cell table with cluster labels
 */
process PIXIE_CELL_CLUSTER {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://bolt3x/attend_image_analysis:pixie'

    publishDir "${params.outdir}/${meta.patient_id}/pixie/cell_clustering", mode: 'copy'

    input:
    tuple val(meta), path(pixel_data_dir), path(cell_table), path(cell_mask), path(cell_params)

    output:
    tuple val(meta), path("cell_output/cluster_counts_size_norm.feather")  , emit: cluster_counts
    tuple val(meta), path("cell_output/cell_som_weights.feather")          , emit: som_weights
    tuple val(meta), path("cell_output/cell_*_cluster_*.csv")              , emit: cluster_profiles
    tuple val(meta), path("cell_output/cell_meta_cluster_mapping.csv")     , emit: cluster_mapping
    tuple val(meta), path("cell_masks/*_cell_mask.tiff")                   , emit: cell_cluster_masks, optional: true
    tuple val(meta), path("cell_output/cell_table_with_clusters.csv")      , emit: cell_table_clustered
    tuple val(meta), path("cell_output/pixie_clusters.geojson")            , emit: geojson
    tuple val(meta), path("cell_output/pixie_clusters.classifications.json"), emit: classifications
    tuple val(meta), path("cell_output/pixie_clusters_mapping.json")       , emit: mapping_json
    path "versions.yml"                                                     , emit: versions
    path("*.size.csv")                                                      , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    # Log input sizes for tracing
    pixel_bytes=\$(du -sb ${pixel_data_dir} 2>/dev/null | cut -f1 || echo 0)
    table_bytes=\$(stat -L --printf="%s" ${cell_table} 2>/dev/null || echo 0)
    mask_bytes=\$(stat -L --printf="%s" ${cell_mask} 2>/dev/null || echo 0)
    total_bytes=\$((pixel_bytes + table_bytes + mask_bytes))
    echo "${task.process},${meta.patient_id},pixel_data+cell_table+mask,\${total_bytes}" > ${meta.patient_id}.PIXIE_CELL_CLUSTER.size.csv

    echo "Sample: ${meta.patient_id}"
    echo "Cell table: ${cell_table}"

    # Create output directories
    mkdir -p cell_output
    mkdir -p cell_masks

    # Run cell clustering
    pixie_cell_cluster.py \\
        --base_dir . \\
        --pixel_output_dir ${pixel_data_dir} \\
        --cell_table_path ${cell_table} \\
        --cell_params_path ${cell_params} \\
        --output_dir cell_output \\
        --pixel_cluster_col ${params.pixie_pixel_cluster_col} \\
        --max_k ${params.pixie_max_k} \\
        --cap ${params.pixie_cap} \\
        --seed ${params.pixie_seed} \\
        --pixel_size ${params.pixel_size} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        ark-analysis: \$(python -c "import ark; print(ark.__version__)" 2>/dev/null || echo "0.6.4")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        pandas: \$(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    mkdir -p cell_output
    mkdir -p cell_masks
    touch cell_output/cluster_counts_size_norm.feather
    touch cell_output/cell_som_weights.feather
    touch cell_output/cell_som_cluster_count_avg.csv
    touch cell_output/cell_meta_cluster_count_avg.csv
    touch cell_output/cell_som_cluster_channel_avg.csv
    touch cell_output/cell_meta_cluster_channel_avg.csv
    touch cell_output/cell_meta_cluster_mapping.csv
    touch cell_output/cell_table_with_clusters.csv
    echo '{"type": "FeatureCollection", "features": []}' > cell_output/pixie_clusters.geojson
    echo '[]' > cell_output/pixie_clusters.classifications.json
    echo '{}' > cell_output/pixie_clusters_mapping.json
    touch cell_masks/${meta.patient_id}_cell_mask.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.PIXIE_CELL_CLUSTER.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        ark-analysis: stub
        numpy: stub
        pandas: stub
    END_VERSIONS
    """
}
