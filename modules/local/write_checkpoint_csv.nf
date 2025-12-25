nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: WRITE_CHECKPOINT_CSV
========================================================================================
    Description:
        Writes checkpoint CSV files for pipeline restartability.
        Creates CSV files with absolute paths to intermediate results that can be
        used to restart the pipeline from a specific step without relying on
        Nextflow's work directory cache.

    Input:
        csv_name: Name of the CSV file (without extension)
        header: CSV header string
        rows: List of row data (each row is a list of values)

    Output:
        csv: The generated CSV file

    Usage:
        Used by subworkflows to create checkpoint files at major pipeline steps
========================================================================================
*/

process WRITE_CHECKPOINT_CSV {
    publishDir "${params.outdir}/csv", mode: 'copy', overwrite: true

    input:
    val(csv_name)
    val(header)
    val(rows)

    output:
    path("${csv_name}.csv"), emit: csv

    script:
    def content = rows.collect { row -> row.join(',') }.join('\n')
    """
    cat > ${csv_name}.csv <<'EOF'
${header}
${content}
EOF
    """
}
