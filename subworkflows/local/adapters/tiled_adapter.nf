/*
========================================================================================
    TILED (STARE) REGISTRATION ADAPTER
========================================================================================
    Registers every moving slide of a patient directly to the fixed reference, in parallel,
    as a per-tile fan-out: TILED_COARSE -> TILED_REG_TILE (one task per tile) -> TILED_SOLVE
    -> TILED_STITCH. Peak memory of each process depends on a parameter (coarse max dim,
    tile + halo, out tile), not on slide size.

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]

    Adapter contract (shared with adapters/valis_adapter.nf; any new adapter must follow it):

        registered          [meta, file]                registered slides (+ passthroughs)
        transform           [patient_id, transform]     one transform object per patient
        transform_by_slide  [meta, transform]           one transform per moving slide
        stage_checkpoint    [patient_id, dir]           intermediate-stage fields
        intrinsic_tre       file                        the method's own TRE estimate, in its
                                                        own format (the reader handles formats)
        size_logs / versions

    An emit the method cannot produce is `Channel.empty()`, never missing, so consumers
    need not know which method ran.
========================================================================================
*/

include { TILED_COARSE   } from '../../../modules/local/tiled_coarse'
include { TILED_REG_TILE } from '../../../modules/local/tiled_reg_tile'
include { TILED_SOLVE    } from '../../../modules/local/tiled_solve'
include { TILED_STITCH   } from '../../../modules/local/tiled_stitch'

// Stable per-moving-slide join key (patient + its channel set).
def slideKey(meta) { "${meta.patient_id}#${meta.channels.toSorted().join('_')}" }

// The control-point gather sizes its groupKey with the tile count so each slide's group closes
// when its own tiles arrive. A missing count must fail: an unsized key waits for the whole channel.
def requireTilesCount(meta) {
    if (meta.tiles_count == null) {
        error "TILED_ADAPTER: tiles_count missing from meta for ${slideKey(meta)} -- cannot size the control-point gather"
    }
    meta.tiles_count
}

// Data rows of a tile-plan CSV (non-blank lines minus the header). This sizes the gather's
// groupKey: too high hangs the group, too low closes it early on missing control points.
def countTileRows(csv) {
    csv.readLines().findAll { it.trim() }.size() - 1
}

// Fails, naming the slide, on an empty tile plan.
def requirePositiveTileCount(n, meta) {
    if (n < 1) error "TILED_COARSE emitted no tiles for ${slideKey(meta)} (n=${n})"
    n
}

workflow TILED_ADAPTER {
    take:
    ch_grouped_meta   // [patient_id, reference_item, all_items]

    main:
    // The reference is the frame and passes through unregistered. is_passthrough tells the
    // checkpoint writer it is published by its producer, not under <pid>/registered/.
    ch_reference = ch_grouped_meta.map { _pid, ref_item, _items ->
        [ref_item[0] + [is_passthrough: true], ref_item[1]]
    }  // [meta, file]

    // One stream item per moving (non-reference) slide: [meta, reference_file, moving_file].
    ch_moving = ch_grouped_meta.flatMap { _pid, ref_item, items ->
        def ref_file = ref_item[1]
        items.findAll { item -> !item[0].is_reference }
             .collect { mov -> tuple(mov[0], ref_file, mov[1]) }
    }

    TILED_COARSE(ch_moving)

    // Tile count per slide, read from the CSV file itself: a splitCsv row cannot see the total.
    ch_tile_counts = TILED_COARSE.out.tiles
        .map { meta, csv -> tuple(slideKey(meta), requirePositiveTileCount(countTileRows(csv), meta)) }

    // Context per slide: meta (+tiles_count) + reference + moving + M0, keyed for the per-tile join.
    ch_ctx = ch_moving
        .map { meta, ref, mov -> tuple(slideKey(meta), meta, ref, mov) }
        .join(TILED_COARSE.out.m0.map { meta, m0 -> tuple(slideKey(meta), m0) }, by: 0)
        .join(ch_tile_counts, by: 0)
        .map { k, meta, ref, mov, m0, n -> tuple(k, meta + [tiles_count: n], ref, mov, m0) }

    // Fan out: one item per tile (splitCsv), attach the slide context.
    ch_tile_items = TILED_COARSE.out.tiles
        .splitCsv(header: true, elem: 1)
        .map { meta, row -> tuple(slideKey(meta), row) }
        .combine(ch_ctx, by: 0)
        .map { _k, row, meta, ref, mov, m0 -> tuple(meta, m0, ref, mov, row) }

    TILED_REG_TILE(ch_tile_items)

    // Gather control points per slide; the sized groupKey lets TILED_SOLVE start as soon as
    // that slide's tiles are done.
    ch_controls = TILED_REG_TILE.out.control
        .map { meta, c -> tuple(groupKey(slideKey(meta), requireTilesCount(meta)), meta, c) }
        .groupTuple(by: 0)
        .map { _k, metas, controls ->
            // groupTuple emits in arrival order; sort the pairs so -resume hashes match.
            // Not groupTuple(sort:), which sorts each list separately and re-pairs them wrongly.
            def paired = [metas, controls].transpose().toSorted { it[1].name }
            tuple(paired[0][0], paired.collect { it[1] })
        }

    ch_solve_in = ch_controls
        .map { meta, controls -> tuple(slideKey(meta), meta, controls) }
        .join(TILED_COARSE.out.m0.map { meta, m0 -> tuple(slideKey(meta), m0) }, by: 0)
        .map { _k, meta, controls, m0 -> tuple(meta, m0, controls) }

    TILED_SOLVE(ch_solve_in)

    ch_stitch_in = TILED_SOLVE.out.manifest
        .map { meta, m -> tuple(slideKey(meta), meta, m) }
        .join(ch_moving.map { meta, _ref, mov -> tuple(slideKey(meta), mov) }, by: 0)
        .map { _k, meta, m, mov -> tuple(meta, m, mov) }

    TILED_STITCH(ch_stitch_in)

    ch_registered_moving = TILED_STITCH.out.registered
    ch_manifest_by_meta  = TILED_SOLVE.out.manifest
    ch_size_logs         = TILED_STITCH.out.size_log
    ch_versions          = TILED_COARSE.out.versions.first()
        .mix(TILED_REG_TILE.out.versions.first())
        .mix(TILED_SOLVE.out.versions.first())
        .mix(TILED_STITCH.out.versions.first())
    ch_intrinsic_tre     = TILED_SOLVE.out.tre.map { _meta, f -> f }

    ch_registered = ch_registered_moving.mix(ch_reference)

    emit:
    registered       = ch_registered
    // STARE manifests keyed by patient (VALIS fills this slot with its registrar pickle);
    // one item per moving slide.
    transform        = ch_manifest_by_meta.map { meta, m -> tuple(meta.patient_id, m) }
    // The same manifests keyed by meta, joined per slide by the reg_qc=2 seg-QC.
    transform_by_slide = ch_manifest_by_meta
    // No pre-micro checkpoint: the tiled method does not compose stages.
    stage_checkpoint = Channel.empty()
    size_logs        = ch_size_logs
    versions         = ch_versions
    // STARE's intrinsic TRE: *_tre.json from bin/utils/tre_report.py (VALIS emits a CSV).
    intrinsic_tre    = ch_intrinsic_tre
}
