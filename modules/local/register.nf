process REGISTER {
    tag "register_all"
    label 'process_high'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://cdgatenbee/valis-wsi:1.0.0' :
        'docker://cdgatenbee/valis-wsi:1.0.0' }"

    input:
    tuple val(reference_filename), path(preproc_files)

    output:
    path "registered_slides/*_registered.ome.tiff", emit: registered_slides
    path "registered_qc"                          , emit: qc, optional: true
    path "versions.yml"                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
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
        --num-features ${num_features} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        valis: \$(python -c "import valis; print(valis.__version__)" 2>/dev/null || echo "unknown")
    END_VERSIONS
    """

    stub:
    """
    mkdir -p registered_slides registered_qc
    touch registered_slides/sample_registered.ome.tiff

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        valis: unknown
    END_VERSIONS
    """
}
