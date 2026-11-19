/*
 * MERGE_QUANT_CSVS - join per-channel intensity CSVs (QUANTIFY) onto the morphology
 * CSV (EXTRACT_CELL_PROPERTIES) by cell label; fails if cells are lost or duplicated.
 */
process MERGE_QUANT_CSVS {
    tag "${meta.patient_id}"
    label 'process_low'

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    tuple val(meta), path(individual_csvs), path(morphology_csv)

    output:
    tuple val(meta), path("merged_quant.csv"), emit: merged_csv
    path "versions.yml"                       , emit: versions
    path("*.size.csv")                        , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // The nuclear marker leads the marker block and is not overwritten on add_cycle.
    // Read via MarkerUtils, which normalises the String/comma-joined param shapes.
    def nuclear_args = "--nuclear-markers ${MarkerUtils.markerList(params.nuclear_markers).join(' ')}"
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ['*_quant.csv', "${morphology_csv}"], "${meta.patient_id}.MERGE_QUANT_CSVS.size.csv")}

    merge_quant_csvs.py \\
        --csv-files ${individual_csvs} \\
        --morphology ${morphology_csv} \\
        --patient-id ${meta.patient_id} \\
        --output merged_quant.csv \\
        ${nuclear_args} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['pandas'], task.container)}
    """

    stub:
    """
    touch merged_quant.csv
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}.MERGE_QUANT_CSVS.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['pandas'], task.container)}
    """
}
