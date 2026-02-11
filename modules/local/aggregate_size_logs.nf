/*
 * AGGREGATE_SIZE_LOGS - Collect and merge all input size logs
 *
 * Aggregates per-task size logs from all processes into a single CSV file
 * for post-run analysis of resource usage vs input size.
 *
 * Input: Collection of size.csv files from all processes
 * Output: Aggregated input_sizes.csv file
 */
process AGGREGATE_SIZE_LOGS {
    tag "aggregate"
    label 'process_single'

    input:
    path(size_csvs)

    output:
    path("input_sizes.csv"), emit: aggregated

    when:
    params.enable_trace

    script:
    """
    echo "process,sample_id,filename,bytes" > input_sizes.csv
    cat ${size_csvs} >> input_sizes.csv

    echo "Aggregated \$(wc -l < input_sizes.csv) size log entries"
    """

    stub:
    """
    echo "process,sample_id,filename,bytes" > input_sizes.csv
    """
}
