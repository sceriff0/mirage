nextflow.enable.dsl = 2

/*
========================================================================================
    REGISTRATION SUBWORKFLOW
========================================================================================
Simplified VALIS image registration workflow (KISS & DRY):

1. COMPUTE_REGISTRAR: Compute transforms and save pickle
2. APPLY_REGISTRATION: Warp slides in parallel using pickle
3. MERGE_REGISTERED: Merge with deduplication
4. GENERATE_QC: Create RGB overlay QC images

Benefits: Parallel processing, memory efficient, proper channel deduplication
========================================================================================
*/

/*
========================================================================================
    PROCESS: COMPUTE_REGISTRAR
========================================================================================
Compute VALIS registrar with all transformation matrices and save as pickle.

Input:
  - path preproc_files: All preprocessed OME-TIFF files (collected)

Output:
  - path registrar.pkl: Pickled VALIS registrar with computed transforms
  - path slide_names.txt: List of slide names for parallel processing
========================================================================================
*/

process COMPUTE_REGISTRAR {
    tag "compute_registrar"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registration", mode: 'copy', pattern: 'registrar.pkl'

    input:
    path preproc_files

    output:
    path "registrar.pkl", emit: registrar
    path "slide_names.txt", emit: slide_names
    path "preprocessed/*", emit: preproc_dir

    script:
    def ref_markers = params.reg_reference_markers ? "--reference-markers ${params.reg_reference_markers.join(' ')}" : ''
    def max_processed_dim = params.reg_max_processed_dim ?: 1800
    def max_non_rigid_dim = params.reg_max_non_rigid_dim ?: 3500
    def micro_reg_fraction = params.reg_micro_reg_fraction ?: 0.25
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

    # Compute registrar and save as pickle
    register_compute.py \\
        --input-dir preprocessed \\
        --output-pickle registrar.pkl \\
        ${ref_markers} \\
        --max-processed-dim ${max_processed_dim} \\
        --max-non-rigid-dim ${max_non_rigid_dim} \\
        --micro-reg-fraction ${micro_reg_fraction} \\
        --num-features ${num_features}

    # Extract slide names for parallel processing
    # Remove path and extension from filenames
    for file in preprocessed/*.ome.tif*; do
        basename "\$file" | sed 's/\\.ome\\.tif.*//' >> slide_names.txt
    done
    """
}

/*
========================================================================================
    PROCESS: APPLY_REGISTRATION
========================================================================================
Apply registration transforms to a single slide using pickled registrar.
Runs in parallel for each slide.
========================================================================================
*/

process APPLY_REGISTRATION {
    tag "${slide_name}"
    label 'process_medium'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registered_slides", mode: 'copy'

    input:
    path registrar_pkl
    val slide_name
    path preproc_dir

    output:
    path "${slide_name}_registered.ome.tif", emit: registered

    script:
    """
    register_apply.py \\
        --registrar-pickle ${registrar_pkl} \\
        --slide-name ${slide_name} \\
        --output-file ${slide_name}_registered.ome.tif
    """
}

/*
========================================================================================
    PROCESS: MERGE_REGISTERED
========================================================================================
Merge all registered slides with channel deduplication.
========================================================================================
*/

process MERGE_REGISTERED {
    tag "merge_registered"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registered", mode: 'copy'

    input:
    path registered_files

    output:
    path "merged_all.ome.tiff", emit: merged

    script:
    """
    merge_registered.py \\
        --input-files ${registered_files} \\
        --output-file merged_all.ome.tiff
    """
}

/*
========================================================================================
    PROCESS: GENERATE_QC
========================================================================================
Generate QC RGB overlay images comparing registered slides to reference.
========================================================================================
*/

process GENERATE_QC {
    tag "generate_qc"
    label 'process_medium'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registration_qc", mode: 'copy'

    input:
    path registrar_pkl
    path registered_files

    output:
    path "qc/*_QC_RGB.tif", emit: qc_images, optional: true

    script:
    """
    generate_registration_qc.py \\
        --registrar-pickle ${registrar_pkl} \\
        --registered-dir . \\
        --output-dir qc
    """
}

/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
Simplified registration workflow following KISS principle.
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_preprocessed  // Channel of preprocessed files

    main:
    // Stage 1: Compute registrar
    COMPUTE_REGISTRAR(ch_preprocessed.collect())

    // Stage 2: Apply registration in parallel
    ch_slide_names = COMPUTE_REGISTRAR.out.slide_names
        .splitText()
        .map { it.trim() }

    APPLY_REGISTRATION(
        COMPUTE_REGISTRAR.out.registrar,
        ch_slide_names,
        COMPUTE_REGISTRAR.out.preproc_dir
    )

    // Stage 3: Merge all registered slides
    ch_registered_collected = APPLY_REGISTRATION.out.registered.collect()
    MERGE_REGISTERED(ch_registered_collected)

    // Stage 4: Generate QC images
    GENERATE_QC(
        COMPUTE_REGISTRAR.out.registrar,
        ch_registered_collected
    )

    emit:
    merged = MERGE_REGISTERED.out.merged
    qc = GENERATE_QC.out.qc_images
}
