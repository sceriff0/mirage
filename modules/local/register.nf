nextflow.enable.dsl = 2

process REGISTER {
    tag "register_all"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registered", mode: 'copy'

    input:
    path preproc_files

    output:
    path "merged/merged_all.ome.tiff", emit: merged
    path "merged_qc"          , emit: qc, optional: true

    script:
    def ref_markers = params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    def max_processed_dim = params.reg_max_processed_dim ?: 1800
    def max_non_rigid_dim = params.reg_max_non_rigid_dim ?: 3500
    def micro_reg_fraction = params.reg_micro_reg_fraction ?: 0.25
    def num_features = params.reg_num_features ?: 5000

    """
    mkdir -p merged merged_qc preprocessed

    # Stage all preprocessed files into a directory
    # Only copy .ome.tif files to avoid processing non-image files
    for file in ${preproc_files}; do
        if [[ "\$file" == *.ome.tif ]] || [[ "\$file" == *.ome.tiff ]]; then
            cp "\$file" preprocessed/
        fi
    done

    register_low_mem.py \\
        --input-dir preprocessed \\
        --out merged/merged_all.ome.tiff \\
        --qc-dir merged_qc \\
        ${ref_markers} \\
        --max-processed-dim ${max_processed_dim} \\
        --max-non-rigid-dim ${max_non_rigid_dim} \\
        --micro-reg-fraction ${micro_reg_fraction} \\
        --num-features ${num_features}
    """
}
