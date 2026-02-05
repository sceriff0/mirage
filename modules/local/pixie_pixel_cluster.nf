nextflow.enable.dsl = 2

/*
 * PIXIE_PIXEL_CLUSTER - Pixel-level clustering using Pixie
 *
 * Performs unsupervised clustering of pixels based on marker expression.
 * Uses Self-Organizing Maps (SOM) followed by consensus clustering.
 *
 * Based on: https://github.com/angelolab/pixie (ark-analysis 0.6.4)
 *
 * Features:
 *   - Automatic FOV tiling for large images (>2048x2048)
 *   - Multiprocessing support for parallel tile processing
 *   - Batch size auto-calculated from allocated CPUs
 *
 * Input:
 *   - Split channel TIFFs (from SPLIT_CHANNELS)
 *   - Cell segmentation mask
 *   - List of channels for clustering
 * Output:
 *   - Pixel cluster data (feather files)
 *   - SOM weights
 *   - Cluster profiles
 *   - Parameters for cell clustering
 *   - Tile positions (if tiling was used)
 */
process PIXIE_PIXEL_CLUSTER {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://bolt3x/attend_image_analysis:pixie'

    publishDir "${params.outdir}/${meta.patient_id}/pixie/pixel_clustering", mode: 'copy'

    input:
    tuple val(meta), path(channel_tiffs), path(cell_mask)
    val(channels)

    output:
    tuple val(meta), path("pixel_output/pixel_mat_data")                  , emit: pixel_data
    tuple val(meta), path("pixel_output/pixel_som_weights.feather")       , emit: som_weights
    tuple val(meta), path("pixel_output/pixel_channel_avg_*.csv")         , emit: cluster_profiles
    tuple val(meta), path("pixel_output/channel_norm_post_rowsum.feather"), emit: norm_vals
    tuple val(meta), path("pixel_output/cell_clustering_params.json")     , emit: cell_params
    tuple val(meta), path("pixel_output/tile_positions.json")             , emit: tile_positions, optional: true
    tuple val(meta), path("pixel_masks/*_pixel_mask.tiff")                , emit: pixel_masks, optional: true
    path "versions.yml"                                                    , emit: versions
    path("*.size.csv")                                                     , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    def fov_name = meta.patient_id
    def channels_arg = channels.join(' ')
    // Calculate batch_size from allocated CPUs (use half to leave headroom)
    def batch_size = params.pixie_batch_size ?: Math.max(1, (task.cpus / 2).intValue())
    def tile_size = params.pixie_tile_size ?: 2048
    def multiprocess_flag = params.pixie_multiprocess != false ? '--multiprocess' : ''
    """
    # Log input sizes for tracing
    total_bytes=0
    for tiff in ${channel_tiffs}; do
        bytes=\$(stat -L --printf="%s" \$tiff 2>/dev/null || echo 0)
        total_bytes=\$((total_bytes + bytes))
    done
    mask_bytes=\$(stat -L --printf="%s" ${cell_mask} 2>/dev/null || echo 0)
    total_bytes=\$((total_bytes + mask_bytes))
    echo "${task.process},${meta.patient_id},channel_tiffs+mask,\${total_bytes}" > ${meta.patient_id}.PIXIE_PIXEL_CLUSTER.size.csv

    echo "Sample: ${meta.patient_id}"
    echo "Channels for clustering: ${channels_arg}"
    echo "Tile size: ${tile_size}"
    echo "Batch size: ${batch_size}"
    echo "Multiprocessing: ${multiprocess_flag ? 'enabled' : 'disabled'}"

    # Create FOV directory structure expected by Pixie
    mkdir -p tiff_dir/${fov_name}
    for tiff in ${channel_tiffs}; do
        ln -s \$PWD/\$tiff tiff_dir/${fov_name}/
    done

    # Link segmentation mask with expected naming
    mkdir -p seg_dir
    ln -s \$PWD/${cell_mask} seg_dir/${fov_name}_cell_mask.tif

    # Run pixel clustering with tiling and multiprocessing support
    pixie_pixel_cluster.py \\
        --tiff_dir tiff_dir \\
        --fov_name ${fov_name} \\
        --seg_dir seg_dir \\
        --seg_suffix _cell_mask.tif \\
        --output_dir . \\
        --channels ${channels_arg} \\
        --blur_factor ${params.pixie_blur_factor} \\
        --subset_proportion ${params.pixie_subset_proportion} \\
        --num_passes ${params.pixie_num_passes} \\
        --max_k ${params.pixie_max_k} \\
        --cap ${params.pixie_cap} \\
        --seed ${params.pixie_seed} \\
        --tile_size ${tile_size} \\
        --batch_size ${batch_size} \\
        ${multiprocess_flag} \\
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
    mkdir -p pixel_output/pixel_mat_data
    mkdir -p pixel_masks
    touch pixel_output/pixel_mat_data/${meta.patient_id}.feather
    touch pixel_output/pixel_som_weights.feather
    touch pixel_output/pixel_channel_avg_som_cluster.csv
    touch pixel_output/pixel_channel_avg_meta_cluster.csv
    touch pixel_output/channel_norm_post_rowsum.feather
    echo '{"fovs": ["${meta.patient_id}"], "channels": [], "is_tiled": false, "original_fov": "${meta.patient_id}"}' > pixel_output/cell_clustering_params.json
    # tile_positions.json is optional - only created when tiling is used
    touch pixel_masks/${meta.patient_id}_pixel_mask.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.PIXIE_PIXEL_CLUSTER.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        ark-analysis: stub
        numpy: stub
        pandas: stub
    END_VERSIONS
    """
}
