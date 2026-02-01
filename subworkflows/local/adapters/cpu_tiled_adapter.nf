nextflow.enable.dsl = 2

/*
========================================================================================
    CPU TILED REGISTRATION ADAPTER
========================================================================================
    Memory-efficient adapter for CPU pairwise registration using tiled processing.

    Two-pass architecture:
    Pass 1 (Affine): COMPUTE_TILE_PLAN → AFFINE_TILE (N parallel) → STITCH_AFFINE
    Pass 2 (Diffeo): DIFFEO_TILE (M parallel) → STITCH_DIFFEO

    Key benefits:
    - Memory per tile: 8-32 GB (vs 128-400 GB monolithic)
    - Parallel execution via Nextflow
    - Independent tile retry on failure

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]
            where reference_item = [meta, file] for the reference image
            and all_items = [[meta1, file1], [meta2, file2], ...] for all images
    Output: Channel of [meta, file] tuples (standard format, including references)
========================================================================================
*/

include { COMPUTE_TILE_PLAN } from '../../../modules/local/compute_tile_plan'
include { AFFINE_TILE       } from '../../../modules/local/affine_tile'
include { STITCH_AFFINE     } from '../../../modules/local/stitch_affine'
include { DIFFEO_TILE       } from '../../../modules/local/diffeo_tile'
include { STITCH_DIFFEO     } from '../../../modules/local/stitch_diffeo'

/*
========================================================================================
    PROCESS: PUBLISH_REFERENCE_CPU_TILED
    Simple pass-through to publish reference images to registered directory
========================================================================================
*/
process PUBLISH_REFERENCE_CPU_TILED {
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

workflow CPU_TILED_ADAPTER {
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
    // PASS 1: AFFINE REGISTRATION
    // ========================================================================

    // Step 1.1: Compute tile plan for each image pair
    COMPUTE_TILE_PLAN(ch_pairs)

    // Step 1.2: Expand to individual affine tile jobs
    // Parse JSON to get tile IDs and create parallel jobs
    ch_affine_tiles = COMPUTE_TILE_PLAN.out.plan
        .flatMap { meta, tile_plan, ref, mov ->
            // Read tile plan JSON
            def planText = tile_plan.text
            def planData = new groovy.json.JsonSlurper().parseText(planText)

            // Create a job for each affine tile
            planData.affine_tiles.collect { tile ->
                tuple(meta, tile.tile_id, tile_plan, ref, mov)
            }
        }

    // Step 1.3: Process each affine tile in parallel
    AFFINE_TILE(ch_affine_tiles)

    // Step 1.4: Collect tiles by meta and stitch
    // Group all tiles for each image, then combine with tile plan
    ch_affine_collected = AFFINE_TILE.out.tile
        .groupTuple(by: 0)
        .map { meta, tile_files ->
            // Flatten in case of nested lists
            def files = tile_files.flatten()
            tuple(meta, files)
        }

    // Join with tile plan
    ch_affine_for_stitch = ch_affine_collected
        .join(
            COMPUTE_TILE_PLAN.out.plan.map { meta, tile_plan, ref, mov ->
                tuple(meta, tile_plan)
            }
        )
        .map { meta, tile_files, tile_plan ->
            tuple(meta, tile_plan, tile_files)
        }

    STITCH_AFFINE(ch_affine_for_stitch)

    // ========================================================================
    // PASS 2: DIFFEOMORPHIC REGISTRATION
    // ========================================================================

    // Step 2.1: Prepare inputs for diffeo stage
    // Need: affine output, tile plan, original reference
    ch_diffeo_prep = STITCH_AFFINE.out.affine
        .join(
            COMPUTE_TILE_PLAN.out.plan.map { meta, tile_plan, ref, mov ->
                tuple(meta, tile_plan, ref, mov)
            }
        )

    // Step 2.2: Expand to individual diffeo tile jobs
    ch_diffeo_tiles = ch_diffeo_prep
        .flatMap { meta, affine_img, tile_plan, ref, mov ->
            // Read tile plan JSON
            def planText = tile_plan.text
            def planData = new groovy.json.JsonSlurper().parseText(planText)

            // Create a job for each diffeo tile
            planData.diffeo_tiles.collect { tile ->
                tuple(meta, tile.tile_id, tile_plan, ref, affine_img)
            }
        }

    // Step 2.3: Process each diffeo tile in parallel
    DIFFEO_TILE(ch_diffeo_tiles)

    // Step 2.4: Collect tiles by meta and stitch
    ch_diffeo_collected = DIFFEO_TILE.out.tile
        .groupTuple(by: 0)
        .map { meta, tile_files ->
            def files = tile_files.flatten()
            tuple(meta, files)
        }

    // Join with tile plan and original moving image
    ch_diffeo_for_stitch = ch_diffeo_collected
        .join(
            COMPUTE_TILE_PLAN.out.plan.map { meta, tile_plan, ref, mov ->
                tuple(meta, tile_plan, mov)
            }
        )
        .map { meta, tile_files, tile_plan, mov ->
            tuple(meta, tile_plan, mov, tile_files)
        }

    STITCH_DIFFEO(ch_diffeo_for_stitch)

    // ========================================================================
    // PUBLISH REFERENCES
    // ========================================================================
    // Reference images don't undergo registration, but need to be published
    // to the registered directory for checkpoint CSV consistency

    ch_references = ch_grouped_meta
        .map { patient_id, ref_item, all_items -> ref_item }

    PUBLISH_REFERENCE_CPU_TILED(ch_references)

    // ========================================================================
    // COMBINE OUTPUTS
    // ========================================================================
    // Combine published references with registered images

    ch_all = PUBLISH_REFERENCE_CPU_TILED.out.published.mix(STITCH_DIFFEO.out.registered)

    // Collect size logs from all tiled processes
    ch_size_logs = Channel.empty()
        .mix(AFFINE_TILE.out.size_log)
        .mix(STITCH_AFFINE.out.size_log)
        .mix(DIFFEO_TILE.out.size_log)
        .mix(STITCH_DIFFEO.out.size_log)

    emit:
    registered = ch_all
    size_logs = ch_size_logs
    // QC generation is now decoupled - handled by GENERATE_REGISTRATION_QC module
}
