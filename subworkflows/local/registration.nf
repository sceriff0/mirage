nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { REGISTER                } from '../../modules/local/register'
include { GPU_REGISTER                      } from '../../modules/local/register_gpu'
include { CPU_REGISTER                      } from '../../modules/local/register_cpu'
include { CPU_REGISTER_MULTIRES             } from '../../modules/local/register_cpu_multires'
include { CPU_REGISTER_CDM                  } from '../../modules/local/register_cpu_cdm'
include { COMPUTE_FEATURES                  } from '../../modules/local/compute_features'
include { ESTIMATE_REG_ERROR                } from '../../modules/local/estimate_registration_error'
include { ESTIMATE_REG_ERROR_SEGMENTATION   } from '../../modules/local/estimate_registration_error_segmentation'

/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
    Description:
        Registers images using VALIS (classic), GPU, CPU, or CPU multi-resolution methods.
        VALIS uses preprocessed (non-padded) images for better compatibility.
        For GPU/CPU methods, finds reference image and creates registration pairs from padded images.

        Additionally computes features before registration and estimates registration error
        after registration using VALIS feature detectors and matchers.

    Input:
        ch_padded: Channel of padded OME-TIFF images (for GPU/CPU methods)
        ch_preprocessed: Channel of preprocessed OME-TIFF images (for VALIS method)
        method: Registration method ('valis', 'gpu', 'cpu', 'cpu_multires', or 'cpu_cdm')
        reference_markers: List of markers to identify reference image

    Output:
        registered: Channel of registered images
        qc: Channel of QC outputs (optional)
        error_metrics: Channel of registration error metrics JSON files
        error_plots: Channel of error distribution plots
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_padded           // Channel of padded images (for GPU/CPU methods)
    ch_preprocessed     // Channel of preprocessed images (for VALIS method)
    method              // Registration method
    reference_markers   // List of markers for reference identification

    main:
    if (method != 'valis') {
        // GPU/CPU registration workflow:
        // Collect padded images and find reference once
        // This ensures deterministic behavior for resume
        ch_padded_collected = ch_padded
            .collect()
            .map { files ->
                // Sort files by name for deterministic ordering
                def sorted_files = files.sort { f -> f.name }

                // Find reference image based on reference_markers
                def reference = sorted_files.find { f ->
                    def filename = f.name.toUpperCase()
                    reference_markers.every { marker ->
                        filename.contains(marker.toUpperCase())
                    }
                }

                // Fallback to first file if no match found
                if (reference == null) {
                    log.warn "No file found with all reference markers ${reference_markers}, using first file"
                    reference = sorted_files[0]
                }

                // Return reference and all files
                return tuple(reference, sorted_files)
            }

        // Create (reference, moving) pairs from padded images
        ch_pairs = ch_padded_collected
            .flatMap { reference, all_files ->
                def moving_files = all_files.findAll { f -> f != reference }
                return moving_files.collect { m -> tuple(reference, m) }
            }

        // STEP 1: Compute features before registration (for error estimation)
        // COMPUTE_FEATURES ( ch_pairs )
        // ch_pre_features = COMPUTE_FEATURES.out.features

        // STEP 2: Perform registration
        if (method == 'gpu') {
            // GPU Registration
            GPU_REGISTER ( ch_pairs )
            ch_registered_moving = GPU_REGISTER.out.registered
            ch_qc = GPU_REGISTER.out.qc
        } else if (method == 'cpu_multires') {
            // CPU Multi-Resolution Registration (coarse → fine affine → diffeo)
            CPU_REGISTER_MULTIRES ( ch_pairs )
            ch_registered_moving = CPU_REGISTER_MULTIRES.out.registered
            ch_qc = CPU_REGISTER_MULTIRES.out.qc
        } else if (method == 'cpu_cdm') {
            // CPU CDM Registration (coarse affine → diffeo → micro affine)
            CPU_REGISTER_CDM ( ch_pairs )
            ch_registered_moving = CPU_REGISTER_CDM.out.registered
            ch_qc = CPU_REGISTER_CDM.out.qc
        } else {
            // CPU Registration (standard 2-stage)
            CPU_REGISTER ( ch_pairs )
            ch_registered_moving = CPU_REGISTER.out.registered
            ch_qc = CPU_REGISTER.out.qc
        }

        // Extract reference from collected channel (no need to recompute)
        ch_reference = ch_padded_collected
            .map { reference, _all_files -> reference }

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

        // Collect preprocessed images and find reference
        ch_preprocessed_collected = ch_preprocessed
            .collect()
            .map { files ->
                def sorted_files = files.sort { f -> f.name }

                def reference = sorted_files.find { f ->
                    def filename = f.name.toUpperCase()
                    reference_markers.every { marker ->
                        filename.contains(marker.toUpperCase())
                    }
                }

                if (reference == null) {
                    log.warn "No file found with all reference markers ${reference_markers}, using first file"
                    reference = sorted_files[0]
                }

                return tuple(reference, sorted_files)
            }

        // Create (reference, moving) pairs for feature computation
        '''ch_valis_pairs = ch_preprocessed_collected
            .flatMap { reference, all_files ->
                def moving_files = all_files.findAll { f -> f != reference }
                return moving_files.collect { m -> tuple(reference, m) }
            }
        '''
        // STEP 1: Compute features before VALIS registration
        // COMPUTE_FEATURES ( ch_valis_pairs )
        // ch_pre_features_valis = COMPUTE_FEATURES.out.features

        // STEP 2: Perform VALIS registration
        REGISTER ( ch_preprocessed.collect() )
        ch_registered_valis = REGISTER.out.registered_slides.flatten()
        ch_qc = channel.empty()

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

    emit:
    registered = ch_registered
    qc = ch_qc
    error_metrics = channel.empty()
    error_plots = channel.empty()
    error_metrics_segmentation = params.enable_segmentation_error != false ? channel.empty() : channel.empty()
    error_overlays = params.enable_segmentation_error != false ? channel.empty() : channel.empty()
}
