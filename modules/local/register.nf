nextflow.enable.dsl = 2

process REGISTER {
    tag "register_all"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered", mode: 'copy'

    input:
    tuple val(reference_filename), path(preproc_files)

    output:
    path "registered_slides/*_registered.ome.tiff", emit: registered_slides
    path "registered_qc"                         , emit: qc, optional: true

    script:
    // Use reference filename if provided, otherwise fall back to legacy reference_markers param
    def ref_arg = reference_filename ? "--reference ${reference_filename}" :
                  params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    def max_processed_dim = params.reg_max_processed_dim ?: 1800
    def max_non_rigid_dim = params.reg_max_non_rigid_dim ?: 3500
    def micro_reg_fraction = params.reg_micro_reg_fraction ?: 0.25
    def num_features = params.reg_num_features ?: 5000

    """
    mkdir -p registered_slides registered_qc preprocessed

    # Stage all preprocessed files into a directory
    # Only copy .ome.tif files to avoid processing non-image files
    for file in ${preproc_files}; do
        if [[ "\$file" == *.ome.tif ]] || [[ "\$file" == *.ome.tiff ]]; then
            mv "\$file" preprocessed/
        fi
    done

    register.py \\
        --input-dir preprocessed \\
        --out registered_slides \\
        --qc-dir registered_qc \\
        ${ref_arg} \\
        --max-processed-dim ${max_processed_dim} \\
        --max-non-rigid-dim ${max_non_rigid_dim} \\
        --micro-reg-fraction ${micro_reg_fraction} \\
        --num-features ${num_features}
    """
}
