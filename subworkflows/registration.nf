nextflow.enable.dsl = 2

/*
========================================================================================
    REGISTRATION SUBWORKFLOW (3-STEP VALIS)
========================================================================================
    Step 1: Compute base registration (rigid + non-rigid) -> pickle
    Step 2: Compute micro-registration -> updated pickle
    Step 3: Warp, merge slides, generate QC
*/

process COMPUTE_REGISTRATION {
    tag "compute_base_registration"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registration/step1", mode: 'copy'

    input:
    path preproc_files

    output:
    path "step1_registrar.pkl", emit: registrar_pkl
    path "preprocessed"        , emit: preprocessed_dir

    script:
    def ref_markers = params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    def max_processed_dim = params.reg_max_processed_dim ?: 1800
    def max_non_rigid_dim = params.reg_max_non_rigid_dim ?: 3500
    def num_features = params.reg_num_features ?: 5000

    """
    mkdir -p preprocessed

    # Stage all preprocessed files into a directory
    # Only copy .ome.tif files to avoid processing non-image files
    for file in ${preproc_files}; do
        if [[ "\$file" == *.ome.tif ]] || [[ "\$file" == *.ome.tiff ]]; then
            cp "\$file" preprocessed/
        fi
    done

    register_step1_compute.py \\
        --input-dir preprocessed \\
        --output-pickle step1_registrar.pkl \\
        ${ref_markers} \\
        --max-processed-dim ${max_processed_dim} \\
        --max-non-rigid-dim ${max_non_rigid_dim} \\
    """
}

process COMPUTE_MICRO_REGISTRATION {
    tag "compute_micro_registration"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registration/step2", mode: 'copy'

    input:
    path registrar_pkl
    path preproc_dir

    output:
    path "step2_registrar.pkl", emit: registrar_pkl

    script:
    def micro_reg_fraction = params.reg_micro_reg_fraction ?: 0.25

    """
    register_step2_micro.py \\
        --input-pickle ${registrar_pkl} \\
        --output-pickle step2_registrar.pkl \\
        --preprocessed-dir ${preproc_dir} \\
        --micro-reg-fraction ${micro_reg_fraction}
    """
}

process WARP_AND_MERGE {
    tag "warp_merge_qc"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registered", mode: 'copy'

    input:
    path registrar_pkl
    path preproc_dir

    output:
    path "merged_all.ome.tiff", emit: merged
    path "qc"                  , emit: qc, optional: true

    script:
    def qc_flag = params.reg_generate_qc ? "--qc-dir qc" : ""

    """
    register_step3_warp.py \\
        --input-pickle ${registrar_pkl} \\
        --output-merged merged_all.ome.tiff \\
        --preprocessed-dir ${preproc_dir} \\
        ${qc_flag}
    """
}

workflow REGISTRATION {
    take:
    preproc_files  // Channel of preprocessed OME-TIFF files

    main:
    // Step 1: Compute base registration
    COMPUTE_REGISTRATION(preproc_files)

    // Step 2: Compute micro-registration
    COMPUTE_MICRO_REGISTRATION(
        COMPUTE_REGISTRATION.out.registrar_pkl,
        COMPUTE_REGISTRATION.out.preprocessed_dir
    )

    // Step 3: Warp, merge, and generate QC
    WARP_AND_MERGE(
        COMPUTE_MICRO_REGISTRATION.out.registrar_pkl,
        COMPUTE_REGISTRATION.out.preprocessed_dir
    )

    emit:
    merged = WARP_AND_MERGE.out.merged
    qc     = WARP_AND_MERGE.out.qc
}
