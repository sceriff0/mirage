/*
 * GENERATE_QC_REPORT - assemble the per-step QC artifacts into a single HTML report.
 */
process GENERATE_QC_REPORT {
    tag "qc_report"
    label 'process_low'

    container "bolt3x/mirage-preprocess:1.0.0"

    input:
    path(preprocess_qc_pngs, stageAs: 'preprocess_qc/*')
    path(registration_qc_pngs, stageAs: 'registration_qc/*')
    // The registration method's own TRE estimate (VALIS *_summary.csv / STARE *_tre.json).
    // Renaming this stageAs directory renames the matching folder inside the published
    // mirage_qc_data_*/ bundle — an accepted, deliberate change of published output.
    path(registration_tre_files, stageAs: 'registration_tre/*')
    path(postprocess_qc_pngs, stageAs: 'postprocess_qc/*')
    path(versions_yml)
    path(run_summary_json)
    path(seg_qc_jsons, stageAs: 'seg_qc/*')
    // Per-cell registration residual CSVs (SEG_QC.out.per_cell). Optional: a run with
    // --start postprocessing has no REGISTRATION output to source them from, so this
    // may stage nothing.
    path(seg_residuals, stageAs: 'seg_residuals/*')

    output:
    path "mirage_qc_report_*.html", emit: report
    path "mirage_qc_data_*/"      , emit: data, optional: true
    path "versions.yml"            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def timestamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date())
    """
    generate_qc_report.py \\
        --preprocess-qc preprocess_qc/ \\
        --registration-qc registration_qc/ \\
        --registration-tre registration_tre/ \\
        --postprocess-qc postprocess_qc/ \\
        --versions ${versions_yml} \\
        --run-summary ${run_summary_json} \\
        --seg-qc seg_qc/ \\
        --seg-residuals seg_residuals/ \\
        --output mirage_qc_report_${timestamp}.html \\
        --data-dir mirage_qc_data_${timestamp}/ \\
        ${args}

    ${ProcessEnvelope.versions(task.process, [], task.container)}
    """

    stub:
    def timestamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date())
    """
    mkdir -p mirage_qc_data_${timestamp}
    touch mirage_qc_report_${timestamp}.html

    ${ProcessEnvelope.versionsStub(task.process, [], task.container)}
    """
}
