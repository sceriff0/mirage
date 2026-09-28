/*
 * CELL_QC - per-patient "QC: ..." columns (docs/outputs.md, "Per-cell QC")
 *
 * Takes MERGE_QUANT_CSVS's table and republishes it, augmented, as THE
 * quantification/merged_quant.csv every downstream step reads. `rounds` is the per-slide
 * manifest built in POSTPROCESSING (round_id, is_reference, markers, and the file names
 * of that slide's retention/residual CSVs); it is rendered to JSON here so the val input
 * hashes deterministically (the caller sorts it by round_id).
 */
process CELL_QC {
    tag "${meta.patient_id}"
    label 'process_low'

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    tuple val(meta), path(merged_csv, stageAs: 'base/merged_quant.csv'), val(rounds), path(retention_csvs, stageAs: 'retention/*'), path(residual_csvs, stageAs: 'residuals/*'), path(prior_rounds, stageAs: 'prior/*')

    output:
    tuple val(meta), path("merged_quant.csv")                     , emit: merged_csv
    tuple val(meta), path("${meta.patient_id}_round_qc.csv")      , emit: round_qc
    tuple val(meta), path("${meta.patient_id}_rounds.json")       , emit: rounds
    path "versions.yml"                                           , emit: versions
    path("*.size.csv")                                            , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def nuclear_args = "--nuclear-markers ${MarkerUtils.markerList(params.nuclear_markers).join(' ')}"
    def prior_arg = prior_rounds ? "--prior-rounds ${prior_rounds instanceof List ? prior_rounds[0] : prior_rounds}" : ''
    def rounds_json = groovy.json.JsonOutput.toJson(rounds)
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ['base/merged_quant.csv'], "${meta.patient_id}.CELL_QC.size.csv")}

    mkdir -p retention residuals
    cat > rounds_in.json <<'ROUNDS_EOF'
${rounds_json}
ROUNDS_EOF

    cell_qc.py \\
        --merged base/merged_quant.csv \\
        --rounds rounds_in.json \\
        --retention-dir retention \\
        --residual-dir residuals \\
        --pixel-size ${meta.pixel_size} \\
        ${nuclear_args} \\
        --patient-id ${meta.patient_id} \\
        ${prior_arg} \\
        --out-merged merged_quant.csv \\
        --out-round-qc ${meta.patient_id}_round_qc.csv \\
        --out-rounds ${meta.patient_id}_rounds.json \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['pandas', 'scipy'], task.container)}
    """

    stub:
    """
    cp base/merged_quant.csv merged_quant.csv
    printf 'label,round_id,markers,nuclear_retention,nuclear_retention_raw,displacement_px,displacement_um,dice\\n' > ${meta.patient_id}_round_qc.csv
    echo '[]' > ${meta.patient_id}_rounds.json
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}.CELL_QC.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['pandas', 'scipy'], task.container)}
    """
}
