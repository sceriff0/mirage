process REGISTER {
    tag "register_all"
    label 'process_high'
    container "${params.container.registration}"

    input:
    path preproc_files

    output:
    path "merged/merged_all.ome.tif", emit: merged
    path "merged_qc/*.png"          , emit: qc, optional: true

    script:
    """
    mkdir -p merged merged_qc
    python3 scripts/register.py \\
        --input-files ${preproc_files.join(' ')} \\
        --out merged/merged_all.ome.tif \\
        --qc-dir merged_qc \\
        || cp ${preproc_files[0]} merged/merged_all.ome.tif
    """
}
