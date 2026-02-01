nextflow.enable.dsl = 2

/*
========================================================================================
    CPU REGISTRATION ADAPTER
========================================================================================
    Adapter for CPU pairwise registration.

    CPU registration is already pairwise, so this adapter mainly handles:
    1. Converting patient-grouped data to pairwise input format
    2. Publishing reference images (which don't undergo registration)
    3. Adding references back to output

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]
            where reference_item = [meta, file] for the reference image
            and all_items = [[meta1, file1], [meta2, file2], ...] for all images
    Output: Channel of [meta, file] tuples (standard format, including references)
========================================================================================
*/

include { CPU_REGISTER } from '../../../modules/local/register_cpu'

/*
========================================================================================
    PROCESS: PUBLISH_REFERENCE_CPU
    Simple pass-through to publish reference images to registered directory
========================================================================================
*/
process PUBLISH_REFERENCE_CPU {
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

workflow CPU_ADAPTER {
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
    // RUN CPU REGISTRATION (Already outputs [meta, file])
    // ========================================================================

    CPU_REGISTER(ch_pairs)

    // ========================================================================
    // PUBLISH REFERENCES
    // ========================================================================
    // Reference images don't undergo registration, but need to be published
    // to the registered directory for checkpoint CSV consistency

    ch_references = ch_grouped_meta
        .map { patient_id, ref_item, all_items -> ref_item }

    PUBLISH_REFERENCE_CPU(ch_references)

    // ========================================================================
    // COMBINE OUTPUTS
    // ========================================================================
    // Combine published references with registered images

    ch_all = PUBLISH_REFERENCE_CPU.out.published.concat(CPU_REGISTER.out.registered)

    // Collect size logs
    ch_size_logs = CPU_REGISTER.out.size_log

    emit:
    registered = ch_all
    size_logs = ch_size_logs
    // QC generation is now decoupled - handled by GENERATE_REGISTRATION_QC module
}
