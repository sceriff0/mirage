/*
========================================================================================
    TILED (STARE) REGISTRATION ADAPTER
========================================================================================
    Converts the patient-grouped structure into the tiled method's per-moving-slide star:
    every moving slide registers directly to the fixed reference (which defines the frame),
    independently and in parallel. Unlike the VALIS adapter there is no batch graph build and
    no per-slide OME-channel re-matching — each task already carries its slide's meta.

    ONE execution shape, same channel contract as VALIS_ADAPTER: a per-TILE Nextflow fan-out —
    TILED_COARSE -> TILED_REG_TILE (one task per tile) -> TILED_SOLVE -> TILED_STITCH. It
    maximises parallelism, and every process' peak memory is a function of a parameter
    (reg_tiled_coarse_max_dim, tile+halo, out_tile) rather than of the slide's dimensions.

    A second, single-task shape used to sit behind a boolean flag. Both the flag and that process
    were removed, because it could not hold that bound — it kept both whole slides, an
    all-channel float32 copy and the full warped output live at once, so its memory scaled with
    the input and had to be budgeted from file size. Keeping an unbounded alternative behind a
    flag was shipping a footgun. See CHANGELOG (Unreleased -> Removed).

    Input:  ch_grouped_meta - Channel of [patient_id, reference_item, all_items]

    THE ADAPTER CONTRACT (identical in adapters/valis_adapter.nf, and binding on any third
    adapter). Every registration adapter takes [patient_id, reference_item, all_items]
    and emits EXACTLY these names:

        registered          [meta, file]                registered slides (+ passthroughs)
        transform           [patient_id, transform]     ONE transform object per patient
        transform_by_slide  [meta, transform]           one transform per MOVING slide
        stage_checkpoint    [patient_id, dir]           intermediate-stage fields
        intrinsic_tre       file                        the method's OWN target-registration
                                                        -error estimate, whatever its format
        size_logs / versions

    `intrinsic_tre` is deliberately NOT named after any one method. Both shipped backends
    estimate a TRE from their own registration -- VALIS a feature-distance CSV, STARE a
    *_tre.json -- and the seam used to call the slot `summary` and then re-emit it as
    `valis_summary`, which pinned one method's name into the artifact vocabulary all the way
    out to the QC report. Formats are NOT normalised here; that is the reader's job.

    A method that produces no artifact for one of these emits `Channel.empty()` for it --
    a NULL OBJECT, never a missing emit and never an error. That is what lets
    REGISTER_PATIENT wire both backends with one short branch, and it is why nothing
    downstream of that branch has to know which method ran. A future adapter inherits the
    rule: declare every name; empty the ones your method cannot produce.
========================================================================================
*/

include { TILED_COARSE   } from '../../../modules/local/tiled_coarse'
include { TILED_REG_TILE } from '../../../modules/local/tiled_reg_tile'
include { TILED_SOLVE    } from '../../../modules/local/tiled_solve'
include { TILED_STITCH   } from '../../../modules/local/tiled_stitch'

// Stable per-moving-slide join key (patient + its channel set).
def slideKey(meta) { "${meta.patient_id}#${meta.channels.toSorted().join('_')}" }

// The control-point gather (below) must size its groupKey with the real tile count so each
// slide's group can close as soon as ITS OWN tiles have arrived, instead of waiting for the
// whole upstream channel to close (every tile of every slide of every patient). Unlike the
// three existing groupKey call sites in this repo, this one must NOT degrade to a bare,
// unsized key when the count is missing -- that silently reverts to the full-barrier behaviour
// this fix removes, with no warning. Fail loudly instead.
def requireTilesCount(meta) {
    if (meta.tiles_count == null) {
        error "TILED_ADAPTER: tiles_count missing from meta for ${slideKey(meta)} -- cannot size the control-point gather"
    }
    meta.tiles_count
}

// Counts a tile-plan CSV's data rows: total non-blank lines minus the header. Extracted (not
// left inline in the ch_tile_counts closure below) so the counting logic itself is unit-testable
// in isolation -- this is the number that ends up sizing the groupKey at the gather, so an
// off-by-one here either hangs the pipeline (count too high, the group never closes) or
// truncates the control points TILED_SOLVE fits its mesh from (count too low, the group closes
// early on a silently mis-registered slide). Neither failure mode looks like a crash, which is
// exactly why the count itself -- not just the null-check in requireTilesCount -- needs direct
// coverage.
def countTileRows(csv) {
    csv.readLines().findAll { it.trim() }.size() - 1
}

// Fails loudly (naming the slide) on an empty tile plan, mirroring requireTilesCount's
// never-silently-degrade contract for the other half of the count derivation.
def requirePositiveTileCount(n, meta) {
    if (n < 1) error "TILED_COARSE emitted no tiles for ${slideKey(meta)} (n=${n})"
    n
}

workflow TILED_ADAPTER {
    take:
    ch_grouped_meta   // [patient_id, reference_item, all_items]

    main:
    // The reference is the frame — it passes through unregistered. is_passthrough says
    // so to registration.nf's checkpoint writer: an unwarped reference is published by
    // whoever produced it, not into <pid>/registered/ where no tiled process writes it.
    // Map addition, never meta.clone() — see the aliasing note at valis_adapter.nf:82-88.
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

    // Tile count per slide, derived from the tile-plan CSV file itself -- NOT from the
    // splitCsv rows below: splitCsv streams one item per row, so the closure that sees a
    // row cannot know how many rows the file has in total. The CSV is tiny (one row per
    // tile, plus a header), so counting lines is cheap.
    ch_tile_counts = TILED_COARSE.out.tiles
        .map { meta, csv -> tuple(slideKey(meta), requirePositiveTileCount(countTileRows(csv), meta)) }

    // Context per slide: meta (+tiles_count) + reference + moving + M0, keyed for the
    // per-tile join. Map addition (never meta.clone()) -- see the aliasing note at
    // valis_adapter.nf:82-88.
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

    // Gather every tile's control point back per slide. groupKey(slideKey, tiles_count)
    // lets a slide's group close -- and TILED_SOLVE start -- as soon as that slide's own
    // tiles have all arrived, instead of the bare groupTuple(by: 0) this replaces, which
    // buffered every key until the WHOLE upstream channel closed (i.e. until every tile of
    // every slide of every patient had finished). requireTilesCount errors rather than
    // silently falling back to an unsized key if the count is ever missing.
    ch_controls = TILED_REG_TILE.out.control
        .map { meta, c -> tuple(groupKey(slideKey(meta), requireTilesCount(meta)), meta, c) }
        .groupTuple(by: 0)
        .map { _k, metas, controls ->
            // CANONICAL ORDER -- same defect and same fix as
            // subworkflows/local/quantify_markers.nf's marker grouping. groupTuple
            // emits in ARRIVAL order, so both the control list reaching TILED_SOLVE
            // (hashed positionally, so -resume missed) and `metas[0]` were whichever
            // TILE finished first. Pair FIRST, then sort the pairs; never
            // `groupTuple(sort:)`, which orders each list independently and re-pairs
            // meta with the wrong control file.
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
    // The STARE transform manifest keyed by patient — the same slot VALIS fills with its
    // registrar pickle. A patient with several moving slides contributes several items here.
    transform        = ch_manifest_by_meta.map { meta, m -> tuple(meta.patient_id, m) }
    // The same manifests keyed by meta. Unlike VALIS, the tiled method DOES have one transform
    // per moving slide, and the reg_qc=2 seg-QC joins them per slide.
    transform_by_slide = ch_manifest_by_meta
    // The tiled method composes no stages destructively, so it needs no pre-micro checkpoint:
    // the null object the contract above requires.
    stage_checkpoint = Channel.empty()
    size_logs        = ch_size_logs
    versions         = ch_versions
    // STARE's intrinsic TRE: *_tre.json from bin/utils/tre_report.py. A DIFFERENT format
    // from VALIS's CSV, on purpose — the channel vocabulary is unified, the file formats
    // are not, and teaching the report reader both shapes is a separate change.
    intrinsic_tre    = ch_intrinsic_tre
}
