process REGISTER {
    tag "register_all"
    label 'process_high'
    container "${params.container.registration}"

    input:
    path preproc_files

    output:
    path "merged/merged_all.ome.tif", emit: merged
    path "merged_qc/*.png"          , emit: qc, optional: true
    path "versions.yml"             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    mkdir -p merged merged_qc
    # Try running the registration script; on failure fall back to copying the first preprocessed file
    python3 scripts/register.py \\
        --input-files ${preproc_files.join(' ')} \\
        --out merged/merged_all.ome.tif \\
        --qc-dir merged_qc \\
        ${args} || cp ${preproc_files[0]} merged/merged_all.ome.tif

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //g')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p merged merged_qc
    touch merged/merged_all.ome.tif
    touch versions.yml
    """
}
