nextflow.enable.dsl = 2

/*
========================================================================================
    VALIS PAIRWISE REGISTRATION ADAPTER
========================================================================================
    Adapter for VALIS pairwise registration.

    Unlike the batch VALIS adapter which processes all patient images together,
    this adapter processes each moving image separately against the reference.
    This is useful when you want VALIS features (SuperPoint/SuperGlue, micro-rigid)
    but don't need the multi-modal batch optimization.

    Benefits over batch VALIS:
    - Lower memory footprint (only 2 images at a time)
    - Simpler per-image parallelization
    - Better for 2-3 stains per patient
    - Same registration quality for pairwise scenarios

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]
            where reference_item = [meta, file] for the reference image
            and all_items = [[meta1, file1], [meta2, file2], ...] for all images
    Output: Channel of [meta, file] tuples (standard format, including references)
========================================================================================
*/

include { REGISTER_VALIS_PAIRS } from '../../../modules/local/register_valis_pairs'

/*
========================================================================================
    PROCESS: PUBLISH_REFERENCE_VALIS_PAIRS
    Simple pass-through to publish reference images to registered directory
========================================================================================
*/
process PUBLISH_REFERENCE_VALIS_PAIRS {
    tag "$meta.id"
    label 'process_single'

    publishDir "${params.outdir}/${meta.patient_id}/registered", mode: 'copy'

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path(image), emit: published
    path "versions.yml"          , emit: versions

    script:
    """
    # File is already staged by Nextflow, publishDir will copy it

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bash: \$(bash --version | head -n1 | sed 's/GNU bash, version //')
    END_VERSIONS
    """

    stub:
    """
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bash: stub
    END_VERSIONS
    """
}

workflow VALIS_PAIRS_ADAPTER {
    take:
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
    // RUN VALIS PAIRWISE REGISTRATION
    // ========================================================================
    // Each pair is registered independently using VALIS
    // - Uses SuperPoint/SuperGlue for feature matching
    // - Applies micro-rigid registration (unless skip_micro_registration=true)
    // - Applies non-rigid optical flow deformation

    REGISTER_VALIS_PAIRS(ch_pairs)

    // ========================================================================
    // PUBLISH REFERENCES
    // ========================================================================
    // Reference images don't undergo registration, but need to be published
    // to the registered directory for checkpoint CSV consistency

    ch_references = ch_grouped_meta
        .map { patient_id, ref_item, all_items -> ref_item }

    PUBLISH_REFERENCE_VALIS_PAIRS(ch_references)

    // ========================================================================
    // COMBINE OUTPUTS
    // ========================================================================
    // Combine published references with registered images

    ch_all = PUBLISH_REFERENCE_VALIS_PAIRS.out.published.concat(REGISTER_VALIS_PAIRS.out.registered)

    // Collect size logs
    ch_size_logs = REGISTER_VALIS_PAIRS.out.size_log

    emit:
    registered = ch_all
    size_logs = ch_size_logs
    // QC generation is now decoupled - handled by GENERATE_REGISTRATION_QC module
}
