nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { GET_IMAGE_DIMS                    } from '../../modules/local/get_image_dims'
include { MAX_DIM                           } from '../../modules/local/max_dim'
include { PAD_IMAGES                        } from '../../modules/local/pad_images'
include { WRITE_CHECKPOINT_CSV              } from '../../modules/local/write_checkpoint_csv'
include { GENERATE_REGISTRATION_QC          } from '../../modules/local/generate_registration_qc'

include { VALIS_ADAPTER                     } from './adapters/valis_adapter'
include { VALIS_PAIRS_ADAPTER               } from './adapters/valis_pairs_adapter'
include { GPU_ADAPTER                       } from './adapters/gpu_adapter'
include { CPU_ADAPTER                       } from './adapters/cpu_adapter'

include { ESTIMATE_FEATURE_DISTANCES        } from '../../modules/local/estimate_feature_distances'


/*
========================================================================================
    SUBWORKFLOW: REGISTRATION
========================================================================================
    Configuration:
        - params.padding: true | false (optional padding per patient)
        - params.skip_registration_qc: true | false (skip QC generation)
        - params.qc_scale_factor: float (QC downsampling factor, default 0.25)
        - params.enable_feature_error: true | false (enable feature-based TRE)
        - params.enable_segmentation_error: true | false (enable segmentation metrics)
        - method: 'valis' | 'valis_pairs' | 'gpu' | 'cpu' (registration method)

    Input:
        ch_preprocessed: Channel of [meta, file] tuples
        method: Registration method name

    Output:
        registered: Channel of [meta, file] tuples (standard format)
        qc: Channel of QC outputs (PNG and TIFF)
        checkpoint_csv: Checkpoint CSV file
        error_metrics: Channel of error estimation outputs (optional)

    QC Generation:
        - Decoupled from registration methods
        - Uses unified GENERATE_REGISTRATION_QC module
        - Compares registered vs reference DAPI channels
        - Outputs: full-res TIFF + downsampled PNG

    Error Estimation (Optional):
        - BEFORE registration: COMPUTE_FEATURES extracts matched keypoints
        - AFTER registration: Two complementary methods:
          1. Feature-based TRE (fast, sparse measurements)
          2. Segmentation-based IoU/Dice (dense, biologically meaningful)
========================================================================================
*/

workflow REGISTRATION {
    take:
    ch_preprocessed
    method

    main:
    // ========================================================================
    // STEP 1: OPTIONAL PADDING (per patient)
    // ========================================================================
    if (params.padding) {
        // Get dimensions for all images
        GET_IMAGE_DIMS(ch_preprocessed)

        // Group by patient and find max dimensions per patient
        ch_grouped_dims = GET_IMAGE_DIMS.out.dims
            .map { meta, dims -> [meta.patient_id, dims] }
            .groupTuple(by: 0)

        MAX_DIM(ch_grouped_dims)

        // MAX_DIM outputs [patient_id, max_dims_file]
        // Combine each individual image with its patient's max_dims_file
        ch_to_pad = ch_preprocessed
            .map { meta, file -> [meta.patient_id, meta, file] }
            .combine(MAX_DIM.out.max_dims_file, by: 0)
            .map { patient_id, meta, file, max_dims -> [meta, file, max_dims] }

        PAD_IMAGES(ch_to_pad)
        ch_images = PAD_IMAGES.out.padded
    } else {
        ch_images = ch_preprocessed
    }

    // ========================================================================
    // STEP 2: GROUP BY PATIENT AND IDENTIFY REFERENCES
    // ========================================================================
    // This is common preparation needed by all methods
    // Output: [patient_id, reference_item, all_items]
    //   where reference_item = [meta, file]
    //   and all_items = [[meta1, file1], [meta2, file2], ...]

    ch_grouped = ch_images
        .map { meta, file -> [meta.patient_id, meta, file] }
        .groupTuple(by: 0)
        .map { patient_id, metas, files ->
            // Combine metas and files into items
            def items = [metas, files].transpose()

            // Find reference image
            def ref = items.find { item -> item[0].is_reference }

            // FIX BUG #3: Make reference fallback configurable instead of silent
            if (!ref) {
                if (params.allow_auto_reference) {
                    log.warn """
                    WARNING: No reference marked for patient ${patient_id}
                    Using first image as reference (allow_auto_reference=true)
                    To make this an error, set allow_auto_reference=false
                    """.stripIndent()
                    ref = items[0]
                } else {
                    throw new Exception("""
                    No reference image found for patient ${patient_id}
                    Fix: Set is_reference=true for one image in your input CSV
                    OR set allow_auto_reference=true to use first image automatically
                    """.stripIndent())
                }
            }

            [patient_id, ref, items]
        }

    // ========================================================================
    // STEP 3: RUN REGISTRATION VIA METHOD-SPECIFIC ADAPTER
    // ========================================================================
    // Each adapter:
    //   - Takes: ch_grouped (patient-grouped structure with references identified)
    //   - Converts to method-specific format
    //   - Runs registration
    //   - Converts output back to [meta, file] standard format

    switch(method) {
        case 'valis':
            VALIS_ADAPTER(ch_grouped)
            ch_registered = VALIS_ADAPTER.out.registered
            break

        case 'valis_pairs':
            VALIS_PAIRS_ADAPTER(ch_grouped)
            ch_registered = VALIS_PAIRS_ADAPTER.out.registered
            break

        case 'gpu':
            GPU_ADAPTER(ch_grouped)
            ch_registered = GPU_ADAPTER.out.registered
            break

        case 'cpu':
            CPU_ADAPTER(ch_grouped)
            ch_registered = CPU_ADAPTER.out.registered
            break

        default:
            error "Invalid registration method: '${method}'. Supported: valis, valis_pairs, gpu, cpu"
    }

    // ========================================================================
    // STEP 3b: GENERATE QC (Method-independent)
    // ========================================================================
    // For each registered image, create QC comparing it to its reference
    // This is now decoupled from the registration method

    // Prepare input for QC: [meta, registered_file, reference_file]
    ch_qc_input = ch_registered
        .branch {
            reference: it[0].is_reference
            moving: !it[0].is_reference
        }

    // Extract references by patient
    ch_references_for_qc = ch_qc_input.reference
        .map { meta, file -> [meta.patient_id, file] }

    // Combine moving images with their patient's reference (1 reference to N moving images)
    ch_for_qc = ch_qc_input.moving
        .map { meta, file -> [meta.patient_id, meta, file] }
        .combine(ch_references_for_qc, by: 0)
        .map { patient_id, meta, registered_file, reference_file ->
            [meta, registered_file, reference_file]
        }

    // Generate QC for all non-reference images
    if (!params.skip_registration_qc) {
        GENERATE_REGISTRATION_QC(ch_for_qc)
        ch_qc = GENERATE_REGISTRATION_QC.out.qc
    } else {
        ch_qc = Channel.empty()
    }

    // ========================================================================
    // STEP 3C: ERROR ESTIMATION (Optional)
    // ========================================================================
    // For each non-reference image, measure quality by comparing:
    //   - reference vs moving (pre-registration)
    //   - reference vs registered (post-registration)

    ch_error_metrics = Channel.empty()

    if (params.enable_feature_error) {
        // For each non-reference image: [meta, reference, moving, registered]
        ch_for_error = ch_registered
            .filter { meta, file -> !meta.is_reference }
            .map { meta, reg_file -> [meta.patient_id, meta.channels.toSorted().join('|'), meta, reg_file] }
            .join(
                ch_images
                    .filter { meta, file -> !meta.is_reference }
                    .map { meta, mov_file -> [meta.patient_id, meta.channels.toSorted().join('|'), mov_file] },
                by: [0, 1]
            )
            .map { patient_id, channels, meta, reg_file, mov_file -> [patient_id, meta, reg_file, mov_file] }
            .combine(
                ch_images
                    .filter { meta, file -> meta.is_reference }
                    .map { meta, ref_file -> [meta.patient_id, ref_file] },
                by: 0
            )
            .map { patient_id, meta, reg_file, mov_file, ref_file ->
                tuple(meta, ref_file, mov_file, reg_file)
            }

        if (params.enable_feature_error) {
            ESTIMATE_FEATURE_DISTANCES(ch_for_error)
            ch_error_metrics = ch_error_metrics.mix(ESTIMATE_FEATURE_DISTANCES.out.distance_metrics)
        }
    }

    // ========================================================================
    // STEP 4: CHECKPOINT
    // ========================================================================
    ch_checkpoint_data = ch_registered
        .map { meta, file ->
            // Construct the path where the file will be published
            // Must match the publishDir configuration in modules.config
            //
            // METHOD-AGNOSTIC APPROACH:
            // Detect if file is in a subdirectory by checking parent directory name length.
            // Nextflow work dirs are 30-char hex hashes (e.g., e6194a65f430c8860ff1f93c4a556c).
            // Real subdirectories (e.g., "registered_slides") have different lengths.
            //
            // - VALIS: work/.../registered_slides/file.tiff → parent="registered_slides" (17 chars)
            // - CPU/GPU: work/.../e6194a65f430c8860ff1f93c4a556c/file.tiff → parent=hash (30 chars)
            // - References: work/.../de93746794b82349b3fde77bf41502/file.tif → parent=hash (30 chars)

            def file_path = file instanceof List ? file[0] : file
            def filename = file_path.name
            def parent_name = file_path.parent?.name ?: ''

            // If parent name is NOT a Nextflow work hash (30 hex chars), it's a real subdirectory
            def is_work_hash = parent_name.length() == 30 && parent_name.matches(/^[0-9a-f]{30}$/)
            def relative_path = is_work_hash ? filename : "${parent_name}/${filename}"

            def published_path = "${params.outdir}/${meta.patient_id}/registered/${relative_path}"
            [meta.patient_id, published_path, meta.is_reference, meta.channels.join('|')]
        }
        .toList()
        .view { data -> "Checkpoint data: $data" }

    WRITE_CHECKPOINT_CSV(
        'registered',
        'patient_id,registered_image,is_reference,channels',
        ch_checkpoint_data
    )

    emit:
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
}
