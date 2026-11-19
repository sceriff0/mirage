/*
 * AGGREGATE_SIZE_LOGS - merge every task's size.csv into one input_sizes.csv
 * (header from ProcessEnvelope.SIZE_LOG_COLUMNS) for resource-vs-input-size analysis.
 */
process AGGREGATE_SIZE_LOGS {
    tag "aggregate"
    label 'process_single'
    // Needs an image with procps: Nextflow's task-metrics wrapper exits without `ps`
    // (a bare ubuntu base fails). The preprocess image is already pulled by other modules.
    container 'bolt3x/mirage-preprocess:1.0.0'

    input:
    path(size_csvs)

    output:
    path("input_sizes.csv"), emit: aggregated
    path "versions.yml"    , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "${ProcessEnvelope.SIZE_LOG_COLUMNS.join(',')}" > input_sizes.csv
    cat ${size_csvs} >> input_sizes.csv

    echo "Aggregated \$(wc -l < input_sizes.csv) size log entries"

    ${ProcessEnvelope.versionsBash(task.process, task.container)}
    """

    stub:
    """
    echo "${ProcessEnvelope.SIZE_LOG_COLUMNS.join(',')}" > input_sizes.csv
    cat ${size_csvs} >> input_sizes.csv

    ${ProcessEnvelope.versionsBashStub(task.process, task.container)}
    """
}
