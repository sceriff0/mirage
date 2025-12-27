nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { REGISTER                          } from '../../modules/local/register'
include { GPU_REGISTER                      } from '../../modules/local/register_gpu'
include { CPU_REGISTER                      } from '../../modules/local/register_cpu'
include { CPU_REGISTER_MULTIRES             } from '../../modules/local/register_cpu_multires'
include { CPU_REGISTER_CDM                  } from '../../modules/local/register_cpu_cdm'
include { MAX_DIM                           } from '../../modules/local/max_dim'
include { PAD_IMAGES                        } from '../../modules/local/pad_images'
include { COMPUTE_FEATURES                  } from '../../modules/local/compute_features'
include { ESTIMATE_REG_ERROR                } from '../../modules/local/estimate_registration_error'
include { ESTIMATE_REG_ERROR_SEGMENTATION   } from '../../modules/local/estimate_registration_error_segmentation'
include { WRITE_CHECKPOINT_CSV              } from '../../modules/local/write_checkpoint_csv'

/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
    Description:
        Registers images using VALIS (classic), GPU, CPU, or CPU multi-resolution methods.
        - VALIS uses preprocessed (non-padded) images directly
        - GPU/CPU methods pad images to uniform size first, then register
        Uses is_reference metadata to identify reference image.

        Additionally computes features before registration and estimates registration error
        after registration using VALIS feature detectors and matchers.

    Input:
        ch_preprocessed: Channel of [meta, file] tuples for preprocessed images
        method: Registration method ('valis', 'gpu', 'cpu', 'cpu_multires', or 'cpu_cdm')

    Output:
        registered: Channel of [meta, file] tuples for registered images
        qc: Channel of QC outputs (optional)
        error_metrics: Channel of registration error metrics JSON files
        error_plots: Channel of error distribution plots
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_preprocessed     // Channel of [meta, file] tuples for preprocessed images
    ch_dims             // Channel of dimension files from preprocessing
    method              // Registration method

    main:
    if (method != 'valis') {
        // GPU/CPU registration workflow:
        // First, pad all preprocessed images to uniform size

        // Compute max dimensions from all dimension files
        // Extract just dimension files (no metadata needed for aggregate operation)
        ch_dims_files = ch_dims.map { meta, dims_file -> dims_file }
        MAX_DIM ( ch_dims_files.collect() )

        // Pad each preprocessed image to max dimensions
        // PAD_IMAGES now accepts [meta, file, max_dims] and preserves metadata
        ch_to_pad = ch_preprocessed
            .map { meta, file -> [meta, file] }
            .combine(MAX_DIM.out.max_dims_file)
            .map { meta, file, max_dims -> [meta, file, max_dims] }

        PAD_IMAGES ( ch_to_pad )

        // PAD_IMAGES now outputs [meta, file] tuples - no reconstruction needed!
        ch_padded = PAD_IMAGES.out.padded

        // Use is_reference metadata to identify reference image
        // Sort by patient_id for deterministic ordering
        ch_padded_collected = ch_padded
            .toList()
            .map { items ->
                if (items.isEmpty()) {
                    error "No preprocessed images found for ${method} registration. Check that preprocessing completed successfully and files exist."
                }

                // Sort by patient_id for deterministic ordering
                def sorted_items = items.sort { it[0].patient_id }

                // Find reference image using is_reference metadata
                def reference_item = sorted_items.find { it[0].is_reference }

                if (reference_item == null) {
                    log.warn "No image with is_reference=true found, using first image"
                    reference_item = sorted_items[0]
                }

                // Validate that the reference item has both metadata and file
                if (reference_item == null || reference_item.size() < 2 || reference_item[1] == null) {
                    error "Invalid reference item: ${reference_item}. Expected [metadata, file] tuple."
                }

                // Return (reference_item, all_items)
                return tuple(reference_item, sorted_items)
            }

        // Create (reference, moving) pairs from padded images
        // Preserve metadata for each moving image
        ch_pairs = ch_padded_collected
            .flatMap { reference_item, all_items ->
                def moving_items = all_items.findAll { !it[0].is_reference }
                return moving_items.collect { moving_item ->
                    tuple(moving_item[0], reference_item[1], moving_item[1])
                }
            }

        // STEP 1: Compute features before registration (for error estimation)
        // COMPUTE_FEATURES ( ch_pairs )
        // ch_pre_features = COMPUTE_FEATURES.out.features

        // STEP 2: Perform registration
        if (method == 'gpu') {
            // GPU Registration
            GPU_REGISTER ( ch_pairs )
            ch_registered_moving_files = GPU_REGISTER.out.registered
            ch_qc = GPU_REGISTER.out.qc
        } else if (method == 'cpu_multires') {
            // CPU Multi-Resolution Registration (coarse → fine affine → diffeo)
            CPU_REGISTER_MULTIRES ( ch_pairs )
            ch_registered_moving_files = CPU_REGISTER_MULTIRES.out.registered
            ch_qc = CPU_REGISTER_MULTIRES.out.qc
        } else if (method == 'cpu_cdm') {
            // CPU CDM Registration (coarse affine → diffeo → micro affine)
            CPU_REGISTER_CDM ( ch_pairs )
            ch_registered_moving_files = CPU_REGISTER_CDM.out.registered
            ch_qc = CPU_REGISTER_CDM.out.qc
        } else {
            // CPU Registration (standard 2-stage)
            CPU_REGISTER ( ch_pairs )
            ch_registered_moving_files = CPU_REGISTER.out.registered
            ch_qc = CPU_REGISTER.out.qc
        }

        // Registered files already have metadata attached by the modules
        // Just flatten and pass through
        ch_registered_moving = ch_registered_moving_files
            .map { meta, file ->
                return tuple(meta, file)
            }

        // Extract reference with metadata
        ch_reference = ch_padded_collected
            .map { reference_item, _all_items -> reference_item }

        // Combine reference and registered moving images (all with metadata)
        ch_registered = ch_reference.concat( ch_registered_moving )

        // STEP 3: Estimate registration error using features
        // Combine registered images with their pre-registration features
        // Need to match registered images with their corresponding feature files by basename
        '''ch_error_input = ch_registered_moving
            .map { reg_file ->
                // Get basename without _registered suffix
                def basename = reg_file.simpleName.replaceAll('_registered$', '')
                return tuple(basename, reg_file)
            }
            .combine(
                ch_pre_features.map { feat_file ->
                    def basename = feat_file.simpleName.replaceAll('_features$', '')
                    return tuple(basename, feat_file)
                },
                by: 0  // Join by basename
            )
            .map { _basename, reg_file, feat_file ->
                // Return (registered, features) tuple (basename no longer needed)
                return tuple(reg_file, feat_file)
            }
            .combine(ch_reference)
            .map { reg_file, feat_file, ref_file ->
                return tuple(ref_file, reg_file, feat_file)
            }

        ESTIMATE_REG_ERROR ( ch_error_input )

        // STEP 4: Estimate registration error using segmentation (optional)
        if (params.enable_segmentation_error != false) {
            ch_segmentation_input = ch_registered_moving
                .combine(ch_reference)
                .map { reg_file, ref_file ->
                    return tuple(ref_file, reg_file)
                }

            ESTIMATE_REG_ERROR_SEGMENTATION ( ch_segmentation_input )
        }
    '''
    } else {
        // Classic VALIS registration: uses preprocessed (non-padded) files
        // Use is_reference metadata to identify reference image (same as GPU/CPU methods)

        // Collect and identify reference using is_reference metadata
        ch_preprocessed_for_valis = ch_preprocessed
            .toList()
            .map { items ->
                if (items.isEmpty()) {
                    error "No preprocessed images found for VALIS registration. Check that preprocessing completed successfully and files exist."
                }

                // Sort by patient_id for deterministic ordering
                def sorted_items = items.sort { it[0].patient_id }

                // Find reference image using is_reference metadata
                def reference_item = sorted_items.find { it[0].is_reference }

                if (reference_item == null) {
                    log.warn "No image with is_reference=true found, using first image"
                    reference_item = sorted_items[0]
                }

                // Validate that the reference item has both metadata and file
                if (reference_item == null || reference_item.size() < 2 || reference_item[1] == null) {
                    error "Invalid reference item: ${reference_item}. Expected [metadata, file] tuple."
                }

                // Extract reference filename and all files for REGISTER process
                def reference_filename = reference_item[1].name
                def all_files = sorted_items.collect { it[1] }

                return tuple(reference_filename, all_files)
            }

        // STEP 1: Compute features before VALIS registration (commented out)
        // COMPUTE_FEATURES ( ch_valis_pairs )
        // ch_pre_features_valis = COMPUTE_FEATURES.out.features

        // STEP 2: Perform VALIS registration
        // Pass reference filename explicitly instead of using reference_markers
        REGISTER ( ch_preprocessed_for_valis )
        ch_registered_valis_files = REGISTER.out.registered_slides.flatten()
        ch_qc = channel.empty()

        // Reconstruct metadata for VALIS registered files
        // Match registered files back to their original metadata from ch_preprocessed (which has [meta, file] tuples)
        ch_registered_valis = ch_registered_valis_files
            .map { file ->
                // Extract base filename to match with original metadata
                def basename = file.name.replaceAll('_corrected_registered', '').replaceAll('.ome.tiff', '')
                return tuple(basename, file)
            }
            .combine(
                ch_preprocessed.map { meta, file ->
                    def basename = file.name.replaceAll('_corrected', '').replaceAll('.ome.tiff', '')
                    return tuple(basename, meta)
                },
                by: 0
            )
            .map { _basename, reg_file, meta ->
                return tuple(meta, reg_file)
            }

        // Extract reference from collected channel
        //ch_reference_valis = ch_preprocessed_collected
        //    .map { reference, _all_files -> reference }

        // STEP 3: Estimate registration error for VALIS
        '''ch_error_input_valis = ch_registered_valis
            .map { reg_file ->
                def basename = reg_file.simpleName.replaceAll('_registered$', '')
                return tuple(basename, reg_file)
            }
            .combine(
                ch_pre_features_valis.map { feat_file ->
                    def basename = feat_file.simpleName.replaceAll('_features$', '')
                    return tuple(basename, feat_file)
                },
                by: 0
            )
            .map { _basename, reg_file, feat_file ->
                return tuple(reg_file, feat_file)
            }
            .combine(ch_reference_valis)
            .map { reg_file, feat_file, ref_file ->
                return tuple(ref_file, reg_file, feat_file)
            }

        ESTIMATE_REG_ERROR ( ch_error_input_valis )

        // STEP 4: Estimate registration error using segmentation (optional)
        if (params.enable_segmentation_error != false) {
            ch_segmentation_input_valis = ch_registered_valis
                .combine(ch_reference_valis)
                .map { reg_file, ref_file ->
                    return tuple(ref_file, reg_file)
                }

            ESTIMATE_REG_ERROR_SEGMENTATION ( ch_segmentation_input_valis )
        }
    '''
        ch_registered = ch_registered_valis
    }

    // Generate checkpoint CSV for restart from registration step
    // ch_registered now contains [meta, file] tuples with full metadata
    ch_checkpoint_data = ch_registered
        .map { meta, file ->
            def abs_path = file.toString()
            [meta.patient_id, abs_path, meta.is_reference, meta.channels.join('|')]
        }
        .collect()

    WRITE_CHECKPOINT_CSV(
        'registered',
        'patient_id,registered_image,is_reference,channels',
        ch_checkpoint_data
    )

    emit:
    registered = ch_registered
    qc = ch_qc
    error_metrics = channel.empty()
    error_plots = channel.empty()
    error_metrics_segmentation = params.enable_segmentation_error != false ? channel.empty() : channel.empty()
    error_overlays = params.enable_segmentation_error != false ? channel.empty() : channel.empty()
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
}
