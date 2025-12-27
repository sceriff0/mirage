nextflow.enable.dsl = 2

/*
========================================================================================
    GPU REGISTRATION ADAPTER
========================================================================================
    Adapter for GPU pairwise registration.

    GPU registration is already pairwise, so this adapter mainly handles:
    1. Converting grouped data to pairwise input format
    2. Adding references back to output

    Input:  Channel of [meta, file] tuples
    Output: Channel of [meta, file] tuples (including references)
========================================================================================
*/

include { GPU_REGISTER } from '../../../modules/local/register_gpu'

workflow GPU_ADAPTER {
    take:
    ch_images         // Channel of [meta, file]
    ch_grouped_meta   // Channel of [patient_id, reference_item, all_items]

    main:
    // ========================================================================
    // CONVERT TO PAIRWISE FORMAT
    // ========================================================================
    // For each patient, create pairs: (moving_meta, reference_file, moving_file)
    // for all non-reference images

    ch_pairs = ch_grouped_meta
        .flatMap { patient_id, ref_item, all_items ->
            def ref_file = ref_item[1]

            all_items
                .findAll { item -> !item[0].is_reference }
                .collect { moving_item ->
                    tuple(moving_item[0], ref_file, moving_item[1])
                }
        }

    // ========================================================================
    // RUN GPU REGISTRATION (Already outputs [meta, file])
    // ========================================================================

    GPU_REGISTER(ch_pairs)

    // ========================================================================
    // ADD REFERENCES BACK
    // ========================================================================
    // GPU_REGISTER only processes non-reference images
    // Need to add reference images back to the output

    ch_references = ch_grouped_meta
        .map { patient_id, ref_item, all_items -> ref_item }

    ch_all = ch_references.concat(GPU_REGISTER.out.registered)

    emit:
    registered = ch_all
    // QC generation is now decoupled - handled by GENERATE_REGISTRATION_QC module
}
