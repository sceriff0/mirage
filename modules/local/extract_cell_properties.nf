/*
 * EXTRACT_CELL_PROPERTIES - per-patient morphology (regionprops) and simplified
 * polygon contours from the cell mask: morphology.csv + contours.json.
 * Computed once here so per-channel QUANTIFY skips regionprops; EXPORT_GEOJSON
 * uses the contours.
 */
process EXTRACT_CELL_PROPERTIES {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    tuple val(meta), path(cell_mask)

    output:
    tuple val(meta), path("morphology.csv") , emit: morphology
    tuple val(meta), path("contours.json")  , emit: contours
    path "versions.yml"                      , emit: versions
    path("*.size.csv")                       , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${cell_mask}"], "${meta.patient_id}.EXTRACT_CELL_PROPERTIES.size.csv")}

    echo "Sample: ${meta.patient_id}"

    extract_cell_properties.py \\
        --mask_file ${cell_mask} \\
        --outdir . \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['skimage'], task.container)}
    """

    stub:
    """
    touch morphology.csv contours.json
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}.EXTRACT_CELL_PROPERTIES.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['skimage'], task.container)}
    """
}
