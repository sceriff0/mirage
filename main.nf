nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_ND2  } from './modules/local/convert_nd2'
include { PREPROCESS   } from './modules/local/preprocess'
include { PAD_IMAGES   } from './modules/local/pad_images'
include { REGISTER     } from './modules/local/register'
include { GPU_REGISTER } from './modules/local/register_gpu'
include { CPU_REGISTER } from './modules/local/register_cpu'
include { MERGE        } from './modules/local/merge'
include { SEGMENT      } from './modules/local/segment'
include { CLASSIFY     } from './modules/local/classify'
include { QUANTIFY     } from './modules/local/quantify'
include { PHENOTYPE    } from './modules/local/phenotype'
include { SAVE_RESULTS } from './modules/local/save_results'


/*
========================================================================================
    RUN MAIN WORKFLOW
========================================================================================
*/

workflow {

    // Validate input parameters
    if (!params.input) {
        error "Please provide an input glob pattern with --input"
    }

    // 1. Create input channel from glob pattern (ND2 files)
    ch_input = channel.fromPath(params.input, checkIfExists: true)

    // 2. MODULE: Convert ND2 to OME-TIFF
    CONVERT_ND2 ( ch_input )

    // 3. MODULE: Preprocess each converted file
    PREPROCESS ( CONVERT_ND2.out.ome_tiff )

    // 4. MODULE: Register using either classic or GPU method
    if (params.registration_method != 'valis') {
        // GPU registration workflow:
        // Step 1: Compute max dimensions from all preprocessed images
        // Using collectFile with storeDir ensures this is cached properly
        ch_dims_file = PREPROCESS.out.dims
            .collectFile(
                name: 'all_dims.txt',
                newLine: true,
                storeDir: "${params.outdir}/${params.id}/${params.registration_method}/metadata"
            )

        // Compute max dimensions once and broadcast as value channel
        ch_max_dims = ch_dims_file
            .map { file ->
                def lines = file.readLines()
                def max_h = 0
                def max_w = 0
                lines.each { line ->
                    def parts = line.split()
                    def h = parts[1].toInteger()
                    def w = parts[2].toInteger()
                    max_h = Math.max(max_h, h)
                    max_w = Math.max(max_w, w)
                }
                return tuple(max_h, max_w)
            }
            .first()  // Convert to value channel for stable caching

        // Step 2: Pad each image in parallel
        ch_to_pad = PREPROCESS.out.preprocessed
            .combine(ch_max_dims)
            .map { file, max_h, max_w -> tuple(file, max_h, max_w) }

        PAD_IMAGES ( ch_to_pad )

        // Step 3: Collect padded images and find reference once
        // This ensures deterministic behavior for resume
        ch_padded_collected = PAD_IMAGES.out.padded
            .collect()
            .map { files ->
                // Sort files by name for deterministic ordering
                def sorted_files = files.sort { f -> f.name }

                // Find reference image based on params.reg_reference_markers
                def reference_markers = params.reg_reference_markers
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

        if (params.registration_method == 'gpu') {
            // Step 4a: GPU Registration
            GPU_REGISTER ( ch_pairs )
            ch_registered = GPU_REGISTER.out.registered
        } else {
            // Step 4b: CPU Registration
            CPU_REGISTER ( ch_pairs )
            ch_registered = CPU_REGISTER.out.registered
        }

        // Extract reference from collected channel (no need to recompute)
        ch_reference = ch_padded_collected
            .map { reference, _all_files -> reference }

        ch_registered = ch_reference.concat( ch_registered )

    } else {
        // Classic registration: collect all preprocessed files
        REGISTER ( PREPROCESS.out.preprocessed.collect() )
        ch_registered = REGISTER.out.registered_slides
    }
    
    // 5. MODULE: Merge registered slides into single multi-channel OME-TIFF
    MERGE ( ch_registered.collect() )

    // 6. MODULE: Segment the merged WSI
    SEGMENT ( MERGE.out.merged )

    // 7. MODULE: Cell classification using DeepCellTypes
    CLASSIFY (
        MERGE.out.merged,
        SEGMENT.out.cell_mask
    )
    
    // 8. MODULE: Quantify marker expression per cell
    QUANTIFY (
        MERGE.out.merged,
        SEGMENT.out.cell_mask
    )

    // 9. MODULE: Phenotype cells based on predefined rules
    PHENOTYPE (
        QUANTIFY.out.csv,
        SEGMENT.out.cell_mask
    )

    // 10. MODULE: Save all results to final output directory
    // Collect all outputs to ensure all processes complete before saving
    ch_all_outputs = channel.empty()
        .mix(
            MERGE.out.merged,
            SEGMENT.out.cell_mask,
            CLASSIFY.out.csv,
            QUANTIFY.out.csv,
            PHENOTYPE.out.csv
        )
        .collect()
        .map { _files ->
            // All files are published under the same parent directory
            return file("${params.outdir}/${params.id}/")
        }

    SAVE_RESULTS (
        ch_all_outputs,
        params.savedir
    )
}
