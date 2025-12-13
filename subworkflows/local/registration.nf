nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { REGISTER              } from '../../modules/local/register'
include { GPU_REGISTER          } from '../../modules/local/register_gpu'
include { CPU_REGISTER          } from '../../modules/local/register_cpu'
include { CPU_REGISTER_MULTIRES } from '../../modules/local/register_cpu_multires'

/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
    Description:
        Registers padded images using VALIS (classic), GPU, CPU, or CPU multi-resolution methods.
        For GPU/CPU methods, finds reference image and creates registration pairs.

    Input:
        ch_padded: Channel of padded OME-TIFF images
        method: Registration method ('valis', 'gpu', 'cpu', or 'cpu_multires')
        reference_markers: List of markers to identify reference image

    Output:
        registered: Channel of registered images
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_padded           // Channel of padded images
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

        if (method == 'gpu') {
            // GPU Registration
            GPU_REGISTER ( ch_pairs )
            ch_registered_moving = GPU_REGISTER.out.registered
            ch_qc = GPU_REGISTER.out.qc
        } else if (method == 'cpu_multires') {
            // CPU Multi-Resolution Registration (coarse-to-fine)
            CPU_REGISTER_MULTIRES ( ch_pairs )
            ch_registered_moving = CPU_REGISTER_MULTIRES.out.registered
            ch_qc = CPU_REGISTER_MULTIRES.out.qc
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

    } else {
        // Classic VALIS registration: uses padded files
        REGISTER ( ch_padded.collect() )
        ch_registered = REGISTER.out.registered_slides
        ch_qc = Channel.empty()
    }

    emit:
    registered = ch_registered
    qc = ch_qc
}
