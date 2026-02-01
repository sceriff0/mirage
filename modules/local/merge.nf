nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MERGE MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Merges single-channel TIFF files (from SPLIT_CHANNELS) into a single multi-channel OME-TIFF.
    DAPI filtering is already handled by SPLIT_CHANNELS (only from reference image).
    Appends segmentation and phenotype masks as additional channels.
    Phenotype mask includes distinct colors for each phenotype for visualization.
----------------------------------------------------------------------------------------
*/

process MERGE {
    tag "${meta.patient_id}"
    label 'process_high'

    container 'docker://bolt3x/attend_image_analysis:merge'

    publishDir "${params.outdir}/${meta.patient_id}/merged", mode: 'copy'

    input:
    tuple val(meta), path(split_channels, stageAs: 'channels/*'), path(seg_mask), path(pheno_mask), path(pheno_mapping)

    output:
    tuple val(meta), path("merged_all.ome.tiff"), emit: merged
    path "versions.yml"                          , emit: versions
    path("*.size.csv")                           , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    # Log input size for tracing (sum of channels/ + masks, -L follows symlinks)
    channels_bytes=\$(du -sb channels/ | cut -f1)
    seg_bytes=\$(stat -L --printf="%s" ${seg_mask})
    pheno_bytes=\$(stat -L --printf="%s" ${pheno_mask})
    total_bytes=\$((channels_bytes + seg_bytes + pheno_bytes))
    echo "${task.process},${meta.patient_id},channels/+masks,\${total_bytes}" > ${meta.patient_id}.MERGE.size.csv

    echo "Sample: ${meta.patient_id}"
    echo "Channels directory: channels/"
    ls -lh channels/

    merge_channels.py \\
        --input-dir channels \\
        --output merged_all.ome.tiff \\
        --segmentation-mask ${seg_mask} \\
        --phenotype-mask ${pheno_mask} \\
        --phenotype-mapping ${pheno_mapping} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}"
    """
    touch merged_all.ome.tiff
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}.MERGE.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        numpy: stub
        tifffile: stub
    END_VERSIONS
    """
}
