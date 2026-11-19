/*
 * EXTRACT_NUCLEI_PROPERTIES - nucleus contours keyed by cell label
 *
 * Runs only with params.quantify_compartments. The cell mask is passed as
 * --reference_mask so contours.json is keyed by cell label and EXPORT_GEOJSON can attach
 * each nucleus to its cell directly. morphology.csv is unused downstream.
 */
process EXTRACT_NUCLEI_PROPERTIES {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    tuple val(meta), path(nuclei_mask), path(cell_mask)

    output:
    tuple val(meta), path("nuclei/morphology.csv") , emit: morphology
    tuple val(meta), path("nuclei/contours.json")  , emit: contours
    path "versions.yml"                      , emit: versions
    path("nuclei/*.size.csv")                , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    # Written into a nuclei/ subdirectory (not '.') so Layout.publishedPath's
    # producerSubdir heuristic (lib/Layout.groovy) can recover the 'nuclei'
    # path segment from the file's own task-directory structure, the same
    # mechanism REGISTER's registered_slides/ and EXPORT_GEOJSON's export/ use.
    # Without this, the checkpoint writer (subworkflows/local/segmentation.nf)
    # has no way to record this file's true published path using only
    # Layout.publishedPath(outdir, pid, 'cell_properties', file). The size log
    # goes in there too (not left at the task root) so the whole process
    # publishes as one unit -- <pid>/cell_properties/nuclei/* -- unchanged from
    # before that fix, rather than splitting its diagnostic artifact out to a
    # different published directory than its real outputs.
    mkdir -p nuclei

    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${nuclei_mask}"], "nuclei/${meta.patient_id}.EXTRACT_NUCLEI_PROPERTIES.size.csv")}

    echo "Sample: ${meta.patient_id}"

    extract_cell_properties.py \\
        --mask_file ${nuclei_mask} \\
        --reference_mask ${cell_mask} \\
        --outdir nuclei \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['skimage'], task.container)}
    """

    stub:
    """
    mkdir -p nuclei
    touch nuclei/morphology.csv nuclei/contours.json
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "nuclei/${meta.patient_id}.EXTRACT_NUCLEI_PROPERTIES.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['skimage'], task.container)}
    """
}
