nextflow.enable.dsl = 2

/*
========================================================================================
    REGISTRATION SUBWORKFLOW
========================================================================================
This subworkflow performs VALIS image registration in three stages:

1. COMPUTE_REGISTRAR: Computes VALIS registration transforms and saves as pickle
2. APPLY_REGISTRATION: Applies transforms to each slide in parallel (including reference)
3. MERGE_REGISTERED: Merges all registered slides with channel deduplication

This architecture enables:
- Parallel processing of individual slides
- Memory-efficient registration
- Proper reference slide handling
- Channel deduplication across slides
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
This process runs in parallel for each slide.

Input:
  - path registrar_pkl: Pickled VALIS registrar
  - val slide_name: Name of slide to register (basename without extension)
  - path preproc_dir: Directory containing preprocessed files

Output:
  - path *_registered.ome.tif: Registered slide as OME-TIFF
  - val slide_name: Slide name (for reference tracking)
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
    val slide_name, emit: slide_name

    script:
    // Check if this is the reference slide
    def ref_flag = params.reg_reference_markers && slide_name.contains(params.reg_reference_markers[0]) ? '--is-reference' : ''

    """
    register_apply.py \\
        --registrar-pickle ${registrar_pkl} \\
        --slide-name ${slide_name} \\
        --output-file ${slide_name}_registered.ome.tif \\
        ${ref_flag}
    """
}

/*
========================================================================================
    PROCESS: MERGE_REGISTERED
========================================================================================
Merge all registered slides into a single multichannel image with channel deduplication.

Input:
  - path registered_files: All registered OME-TIFF files (collected)
  - val ref_slide_name: Name of reference slide (optional)

Output:
  - path merged_all.ome.tiff: Final merged multichannel image
========================================================================================
*/

process MERGE_REGISTERED {
    tag "merge_registered"
    label 'process_high'
    container "${params.container.registration}"

    publishDir "${params.outdir}/registered", mode: 'copy'

    input:
    path registered_files
    val ref_slide_name

    output:
    path "merged_all.ome.tiff", emit: merged

    script:
    def ref_flag = ref_slide_name ? "--reference-slide ${ref_slide_name}" : ''

    """
    merge_registered.py \\
        --input-files ${registered_files} \\
        --output-file merged_all.ome.tiff \\
        ${ref_flag}
    """
}

/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
Main registration subworkflow that orchestrates the three stages.

Input:
  - ch_preprocessed: Channel of preprocessed OME-TIFF files

Output:
  - merged: Merged multichannel registered image
  - ref_slide: Reference slide (for segmentation)
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_preprocessed  // Channel of preprocessed files

    main:
    // Stage 1: Compute registrar and save as pickle
    // Collect all preprocessed files for registration
    ch_preproc_collected = ch_preprocessed.collect()

    COMPUTE_REGISTRAR(ch_preproc_collected)

    // Stage 2: Apply registration to each slide in parallel
    // Create channel from slide names (one per slide)
    ch_slide_names = COMPUTE_REGISTRAR.out.slide_names
        .splitText()
        .map { it.trim() }

    // Apply registration to each slide in parallel
    // Pass registrar pickle, slide name, and preprocessed directory
    APPLY_REGISTRATION(
        COMPUTE_REGISTRAR.out.registrar,
        ch_slide_names,
        COMPUTE_REGISTRAR.out.preproc_dir
    )

    // Identify reference slide
    // Extract reference slide name from slide names
    ch_ref_slide = APPLY_REGISTRATION.out.slide_name
        .filter { slide_name ->
            params.reg_reference_markers &&
            params.reg_reference_markers.any { marker -> slide_name.contains(marker) }
        }
        .first()
        .ifEmpty('') // Default to empty if no reference found

    // Get reference slide registered file
    ch_ref_slide_file = APPLY_REGISTRATION.out.registered
        .filter { file ->
            def slide_name = file.baseName.replace('_registered', '')
            params.reg_reference_markers &&
            params.reg_reference_markers.any { marker -> slide_name.contains(marker) }
        }
        .first()

    // Stage 3: Merge all registered slides with deduplication
    // Collect all registered slides
    ch_registered_collected = APPLY_REGISTRATION.out.registered.collect()

    MERGE_REGISTERED(
        ch_registered_collected,
        ch_ref_slide
    )

    emit:
    merged = MERGE_REGISTERED.out.merged
    ref_slide = ch_ref_slide_file
}
