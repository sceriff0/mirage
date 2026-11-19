/*
========================================================================================
    SUBWORKFLOW: ADD_CYCLE  (incremental cyclic-IF)
========================================================================================
    Folds a NEW imaging cycle into a prior completed patient result. Reuses the
    prior reference (registration target), the prior segmentation cell + nuclei
    masks (no SEGMENT), and the prior merged quantification table (base).
    Recomputes only: preprocessing + registration of the new slide, quantification
    of the new markers, and a wholesale regenerate of cells.geojson + the pyramid
    over the COMBINED channel/measurement set.

    Assumes true cyclic IF: the same physical section re-imaged, so the prior
    masks' cell labels align with the newly registered cycle. Registration-drift
    QC surfaces misalignment; it never blocks scheduling, but a failed QC task
    fails the run (retry-then-fail in conf/modules.config).
      - reg_qc >= 1: DAPI-overlay image QC (GENERATE_REGISTRATION_QC)
      - reg_qc >= 2: + staged seg-overlap QC (SEG_QC) — per-pair IoU and centroid
        residual at each registration stage, using the VALIS registrar pickle from
        the 2-node graph and REGISTER's pre-micro stage checkpoint.
        See docs/registration_qc.md.

    Registration is VALIS only.

    Per-compartment (expanded) quantification is supported: under
    --quantify_compartments, the reused nuclei mask feeds QUANTIFY and
    EXTRACT_NUCLEI_PROPERTIES supplies nucleus contours for EXPORT_GEOJSON.
========================================================================================
*/

include { PREPROCESSING            } from './preprocess'
include { REGISTER_PATIENT         } from './register_patient'
include { EXTRACT_MASK_SERIES      } from '../../modules/local/extract_mask_series'
include { EXTRACT_CELL_PROPERTIES  } from '../../modules/local/extract_cell_properties'
include { EXTRACT_NUCLEI_PROPERTIES } from '../../modules/local/extract_nuclei_properties'
include { SPLIT_CHANNELS           } from '../../modules/local/split_channels'
include { SPLIT_CHANNELS as SPLIT_PRIOR_PYRAMID } from '../../modules/local/split_channels'
include { MERGE_QUANT_CSVS         } from '../../modules/local/merge_quant_csvs'
include { NUCLEAR_RETENTION        } from '../../modules/local/nuclear_retention'
include { CELL_QC                  } from '../../modules/local/cell_qc'
include { GENERATE_REGISTRATION_QC } from '../../modules/local/generate_registration_qc'
include { SEG_QC                   } from './seg_qc'
// Shared with postprocess.nf. groupTiffsByPatient is a plain function.
include { QUANTIFY_MARKERS; groupTiffsByPatient } from './quantify_markers'
include { ASSEMBLE_EXPORT          } from './assemble_export'
// Writing csv/postprocessed.csv makes this --outdir usable as the next cycle's --prior_outdir.
include { POSTPROCESSED_CHECKPOINT } from './postprocessed_checkpoint'

workflow ADD_CYCLE {
    take:
    ch_new_input     // [meta, raw_file]  (new cycle slides, is_reference=false)
    compartment_mode // ParamUtils.compartmentMode(params), resolved in workflows/mirage.nf

    main:
    // ------------------------------------------------------------------ //
    // 0. PRIOR ASSETS — everything this subworkflow reuses from the prior
    //    completed run, rebuilt from that run's checkpoint CSVs.
    // ------------------------------------------------------------------ //
    // Per-patient payload, keyed by a bare patient_id for `by: 0`:
    //
    //   [patient_id, [ ref_channels : List<String>  channels of the frozen reference
    //                , ref_image    : path          the frozen registration target
    //                , base_csv     : path          prior merged quantification table
    //                , cell_mask    : path          prior segmentation cell mask
    //                , nuclei_mask  : path          prior segmentation nuclei mask
    //                , pyramid      : path          prior combined pyramid ]]
    //
    // The prerequisites are validated in workflows/mirage.nf before this runs.

    // Fail if the checkpoint schema drifts: splitCsv would read a lost column as empty.
    Checkpoint.requireColumns(Layout.REGISTERED,
        ['patient_id', 'registered_image', 'is_reference', 'channels'])
    ch_prior_ref = Channel
        .fromPath(Layout.checkpointCsv(params.prior_outdir, Layout.REGISTERED), checkIfExists: true)
        .splitCsv(header: true)
        .filter { row -> row.is_reference?.toLowerCase() == 'true' }
        .map { row ->
            def chans = (row.channels ?: '').split('\\|').collect { it.trim() }.findAll { it }
            [row.patient_id, chans, file(row.registered_image)]
        }

    // The cell_mask column is read but unused: masks are re-extracted from the pyramid below.
    Checkpoint.requireColumns(Layout.POSTPROCESSED,
        ['patient_id', 'merged_csv', 'cell_mask', 'pyramid'])
    ch_prior_rows = Channel
        .fromPath(Layout.checkpointCsv(params.prior_outdir, Layout.POSTPROCESSED), checkIfExists: true)
        .splitCsv(header: true)
        .map { row -> [row.patient_id, file(row.merged_csv), file(row.cell_mask), file(row.pyramid)] }

    // Extract masks from the prior pyramid's Image:1 series (fast-fails in the
    // process if the prior run was not embed_masks+expanded+compartment).
    EXTRACT_MASK_SERIES(ch_prior_rows.map { pid, _merged_csv, _cell_mask, pyramid -> [[patient_id: pid, id: pid], pyramid] })
    ch_extracted_masks = EXTRACT_MASK_SERIES.out.cell_mask.map { m, f -> [m.patient_id, f] }
        .join(EXTRACT_MASK_SERIES.out.nuclei_mask.map { m, f -> [m.patient_id, f] }, by: 0)

    ch_prior_assets = ch_prior_ref
        .join(ch_prior_rows.map { pid, merged_csv, _cell_mask, pyramid -> [pid, merged_csv, pyramid] }, by: 0)
        .join(ch_extracted_masks, by: 0)
        .map { pid, ref_channels, ref_image, base_csv, pyramid, cell_mask, nuclei_mask ->
            [pid, [
                ref_channels: ref_channels,
                ref_image   : ref_image,
                base_csv    : base_csv,
                cell_mask   : cell_mask,
                nuclei_mask : nuclei_mask,
                pyramid     : pyramid,
            ]]
        }

    // ------------------------------------------------------------------ //
    // 1. PREPROCESS the new cycle (mirrors cycle-1 illumination/prep)
    // ------------------------------------------------------------------ //
    PREPROCESSING(ch_new_input)
    ch_new_pre = PREPROCESSING.out.preprocessed   // [meta, file]

    // ------------------------------------------------------------------ //
    // 2. REGISTER new cycle -> frozen prior reference (2-node VALIS graph)
    // ------------------------------------------------------------------ //
    // [patient_id, ref_item, all_items] with the prior reference as a fixed frame;
    // only the new cycle's output is kept.
    ch_ref_item = ch_prior_assets.map { pid, prior ->
        def ref_meta = [patient_id: pid, id: "${pid}_reference", is_reference: true, channels: prior.ref_channels]
        [pid, [ref_meta, prior.ref_image]]
    }
    ch_grouped = ch_new_pre
        .map { meta, file -> [meta.patient_id, [meta + [is_reference: false], file]] }
        .combine(ch_ref_item, by: 0)
        .map { pid, new_item, ref_item ->
            [pid, ref_item, [ref_item, new_item]]   // all_items = reference + new cycle
        }

    // 'valis' is a literal: workflows/mirage.nf rejects tiled in add_cycle mode.
    // REGISTER_PATIENT also writes csv/registered.csv including the reference row, which
    // the next add_cycle reads back. Groups here always have two slides.
    REGISTER_PATIENT(ch_grouped, 'valis')
    ch_adapter_registered = REGISTER_PATIENT.out.registered
    ch_adapter_versions   = REGISTER_PATIENT.out.versions
    ch_transform          = REGISTER_PATIENT.out.transform

    // Keep only the newly registered cycle (drop the reference passthrough).
    ch_new_registered = ch_adapter_registered.filter { meta, _f -> !meta.is_reference }

    // ------------------------------------------------------------------ //
    // 3. REGISTRATION-DRIFT QC (non-gating for scheduling; retry-then-fail on error)
    // ------------------------------------------------------------------ //
    ch_qc            = Channel.empty()
    ch_seg_qc        = Channel.empty()
    // Per-cell registration residuals (final stage).
    ch_seg_residuals = Channel.empty()
    def reg_qc_level = ParamUtils.regQcLevel(params)

    def do_seg_qc = reg_qc_level >= 2

    // Level >= 1: the new cycle's native and registered images against the prior reference.
    // Joined on meta.id, not the meta map, which breaks on any `meta + [k: v]`.
    if (reg_qc_level >= 1) {
        ch_for_qc = ch_new_registered
            .map { meta, f -> [meta.id, meta, f] }
            .join(ch_new_pre.map { meta, f -> [meta.id, f] }, by: 0)
            .map { _id, meta, reg_f, nat_f -> [meta.patient_id, meta, reg_f, nat_f] }
            .combine(ch_prior_assets.map { pid, prior -> [pid, prior.ref_image] }, by: 0)
            .map { _pid, meta, reg_f, nat_f, ref_f -> [meta, reg_f, nat_f, ref_f] }
        GENERATE_REGISTRATION_QC(ch_for_qc)
        ch_qc = GENERATE_REGISTRATION_QC.out.qc
    }

    // Level >= 2: seg-overlap QC on the native reference + new-cycle images, warped through
    // the VALIS registrar. transform_by_slide is empty under VALIS.
    ch_seg_qc_size_log = Channel.empty()
    ch_seg_qc_versions = Channel.empty()
    if (do_seg_qc) {
        ch_native = ch_new_pre
            .map { meta, f -> [meta + [is_reference: false], f] }
            .mix(ch_prior_assets.map { pid, prior ->
                [[patient_id: pid, id: "${pid}_reference", is_reference: true, channels: prior.ref_channels], prior.ref_image]
            })

        SEG_QC(ch_native, ch_transform, REGISTER_PATIENT.out.stage_checkpoint,
               REGISTER_PATIENT.out.transform_by_slide, 'valis')
        ch_seg_qc          = SEG_QC.out.metrics
        ch_seg_residuals   = SEG_QC.out.per_cell
        ch_seg_qc_size_log = SEG_QC.out.size_log
        ch_seg_qc_versions = SEG_QC.out.versions
    }

    // ------------------------------------------------------------------ //
    // 4. REUSE prior masks -> cell contours (+ nucleus contours if compartments)
    //    Masks are unchanged, so cell labels are identical.
    // ------------------------------------------------------------------ //
    ch_prior_cell_mask = ch_prior_assets.map { pid, prior ->
        [[patient_id: pid, id: pid, is_reference: true], prior.cell_mask]
    }
    EXTRACT_CELL_PROPERTIES(ch_prior_cell_mask)
    ch_contours = EXTRACT_CELL_PROPERTIES.out.contours.map { meta, j -> [meta.patient_id, j] }

    // Nucleus contours (re-keyed to cell labels) — only when compartments enabled.
    ch_nucleus_contours = Channel.empty()
    if (compartment_mode.compartments) {
        ch_nuclei_props_in = ch_prior_assets.map { pid, prior ->
            [[patient_id: pid, id: pid], prior.nuclei_mask, prior.cell_mask]
        }
        EXTRACT_NUCLEI_PROPERTIES(ch_nuclei_props_in)
        ch_nucleus_contours = EXTRACT_NUCLEI_PROPERTIES.out.contours.map { meta, j -> [meta.patient_id, j] }
    }

    // ------------------------------------------------------------------ //
    // 5. SPLIT new registered cycle -> per-marker single-channel TIFFs
    // ------------------------------------------------------------------ //
    // INPUT_CHECK resolved keep_channels within this sheet only, which has no reference row,
    // so the new cycle would claim the nuclear channel and replace the reference's DAPI in
    // the pyramid. Subtract what the prior reference already carries. channels_count is
    // not adjusted: step 9 over-counts on purpose.
    def prior_ref_channels = CsvUtils.referenceChannelsPerPatient(
        Layout.checkpointCsv(params.prior_outdir, Layout.REGISTERED))

    ch_new_for_split = ch_new_registered.map { meta, f ->
        def alreadyInPrior = ((prior_ref_channels[meta.patient_id] ?: [])
            .collect { it.toString().toUpperCase() }) as Set
        // Not `?:`: an empty keep_channels is an answer, and `?:` would restore the full panel.
        def declared = meta.keep_channels != null ? meta.keep_channels : (meta.channels ?: [])
        def keep = declared.findAll { !alreadyInPrior.contains(it.toString().toUpperCase()) }
        [meta + [keep_channels: keep], f, false]
    }
    // A slide whose keep-set is now empty adds nothing; filter it out, as postprocess.nf does.
    // Left in, SPLIT_CHANNELS would get no --keep-channels and re-emit prior markers.
    SPLIT_CHANNELS(
        ch_new_for_split.filter { meta, _f, _is_ref ->
            if (meta.keep_channels.isEmpty()) {
                log.warn "ADD_CYCLE(${meta.patient_id}): new-cycle slide ${meta.id} adds no markers the " +
                         "prior run's reference does not already carry. Not splitting it; the prior " +
                         "pyramid's copies are kept."
                return false
            }
            return true
        }
    )

    // ------------------------------------------------------------------ //
    // 6. QUANTIFY new markers against the reused masks (QUANTIFY_MARKERS,
    //    shared with postprocess.nf).
    // ------------------------------------------------------------------ //
    ch_masks = ch_prior_assets.map { pid, prior ->
        [pid, prior.cell_mask, prior.nuclei_mask]
    }
    QUANTIFY_MARKERS(SPLIT_CHANNELS.out.channels, ch_masks)

    // ------------------------------------------------------------------ //
    // 7. MERGE new marker CSVs onto the prior merged table (base) by label
    // ------------------------------------------------------------------ //
    // The third MERGE_QUANT_CSVS slot is the prior base table (no morphology join).
    // pixel_size is carried explicitly because the meta below is a fresh literal.
    ch_new_quant_grouped = QUANTIFY_MARKERS.out.grouped_csv
        .map { meta, csvs -> [meta.patient_id, meta.pixel_size, csvs] }
    ch_base = ch_prior_assets.map { pid, prior -> [pid, prior.base_csv] }
    ch_for_merge = ch_new_quant_grouped
        .combine(ch_base, by: 0)
        .map { pid, pixel_size, csvs, base_csv -> [[patient_id: pid, id: pid, pixel_size: pixel_size], csvs, base_csv] }
    MERGE_QUANT_CSVS(ch_for_merge)

    // ------------------------------------------------------------------ //
    // 7b. CELL QC — the new rounds' QC columns folded onto the prior table
    // ------------------------------------------------------------------ //
    // As in postprocess.nf. Prior rounds' QC columns come through the base table, and the
    // prior <pid>_rounds.json is passed so the manifest is prior + new (new wins a shared
    // round_id). NUCLEAR_RETENTION runs on the same slides SPLIT_CHANNELS gets.
    def contributes = { meta -> !(meta.keep_channels != null && meta.keep_channels.isEmpty()) }

    ch_retention_in = ch_new_for_split
        .filter { meta, _f, _is_ref -> contributes.call(meta) }
        .map { meta, img, _is_ref -> [meta.patient_id, meta, img] }
        .combine(ch_masks, by: 0)
        .map { _pid, meta, img, cmask, nmask -> [meta, img, cmask, nmask] }
    NUCLEAR_RETENTION(ch_retention_in)

    // Keyed by [patient_id, id]: ids are unique only within a patient.
    ch_round_rows = SPLIT_CHANNELS.out.channels
        .map { meta, tiffs ->
            def list = tiffs instanceof List ? tiffs : [tiffs]
            [[meta.patient_id, meta.id], meta.patient_id, meta.is_reference as boolean, list.collect { it.baseName }.toSorted()]
        }
        .join(NUCLEAR_RETENTION.out.csv.map { meta, csv -> [[meta.patient_id, meta.id], csv] }, by: 0, remainder: true)
        .join(ch_seg_residuals.map { meta, csv -> [[meta.patient_id, meta.id], csv] }, by: 0, remainder: true)
        .filter { row -> row[1] != null }     // a residual whose slide never split (keep-set empty)
        .map { key, pid, is_ref, markers, ret_csv, res_csv ->
            [pid, [round_id: key[1], is_reference: is_ref, markers: markers,
                   retention_csv: ret_csv ? ret_csv.name : null,
                   residual_csv : res_csv ? res_csv.name : null],
             ret_csv, res_csv]
        }
        // A cross-patient barrier: the remainder joins release only on channel close.
        .groupTuple(by: 0)
        .map { pid, entries, ret_csvs, res_csvs ->
            // Sort for -resume; file lists pair with entries by name.
            [pid,
             entries.toSorted { it.round_id },
             ret_csvs.findAll { it != null }.toSorted { it.name },
             res_csvs.findAll { it != null }.toSorted { it.name }]
        }

    // The prior run's rounds manifest, or [] if it has none.
    ch_cell_qc_in = MERGE_QUANT_CSVS.out.merged_csv
        .map { meta, csv -> [meta.patient_id, meta, csv] }
        .join(ch_round_rows, by: 0, remainder: true)
        .filter { row -> row[1] != null }
        .map { pid, meta, csv, rounds, rets, ress ->
            def prior_manifest = file("${Layout.patientDir(params.prior_outdir, pid, 'quantification')}/${pid}_rounds.json")
            [meta, csv, rounds ?: [], rets ?: [], ress ?: [], prior_manifest.exists() ? prior_manifest : []]
        }
    CELL_QC(ch_cell_qc_in)
    ch_merged_with_qc = CELL_QC.out.merged_csv

    // ------------------------------------------------------------------ //
    // 8. EXPORT complete cells.geojson from the COMBINED table.
    //    nucleus slot: real nucleus contours when compartments enabled,
    //    else the cell contours (harmless placeholder — EXPORT_GEOJSON only
    //    passes --nucleus_contours_json under params.quantify_compartments).
    // ------------------------------------------------------------------ //
    ch_nuc_for_export = compartment_mode.compartments ? ch_nucleus_contours : ch_contours

    // ------------------------------------------------------------------ //
    // 9. REBUILD complete pyramid: recover prior channels from the prior
    //    pyramid (is_reference=true keeps ref DAPI + all old markers), then
    //    merge with the new cycle's channels.
    // ------------------------------------------------------------------ //
    ch_prior_pyramid = ch_prior_assets.map { pid, prior ->
        [[patient_id: pid, id: pid, is_reference: true, channels: []], prior.pyramid, true]
    }
    SPLIT_PRIOR_PYRAMID(ch_prior_pyramid)

    // Deterministic new-wins dedup on a marker name: new channels priority 0, prior 1, keep
    // the lowest. A .mix + .unique would depend on scheduling.
    ch_new_tagged = SPLIT_CHANNELS.out.channels.flatMap { meta, tiffs ->
        (tiffs instanceof List ? tiffs : [tiffs]).collect { tiff -> [[meta.patient_id, tiff.baseName], 0, tiff] }
    }
    ch_prior_tagged = SPLIT_PRIOR_PYRAMID.out.channels.flatMap { meta, tiffs ->
        (tiffs instanceof List ? tiffs : [tiffs]).collect { tiff -> [[meta.patient_id, tiff.baseName], 1, tiff] }
    }
    ch_deduped_channels = ch_new_tagged.mix(ch_prior_tagged)
        .groupTuple(by: 0)   // [[pid, marker], [priorities...], [tiffs...]]
        .map { key, prios, tiffs ->
            def winner = [prios, tiffs].transpose().sort { a, b -> a[0] <=> b[0] }.first()
            [key[0], winner[1]]   // [pid, winning_tiff] — lowest priority (new) wins
        }

    // Per-patient channel count for groupTiffsByPatient: new + prior, not subtracting
    // collisions, because over-counting only closes the group late, and an under-count aborts.
    // mix + groupTuple + sum rather than join, so a patient missing on one side still
    // gets a count instead of disappearing.
    ch_channel_counts = SPLIT_CHANNELS.out.channels
        .map { meta, tiffs -> [meta.patient_id, (tiffs instanceof List ? tiffs.size() : 1)] }
        .mix(
            SPLIT_PRIOR_PYRAMID.out.channels
                .map { meta, tiffs -> [meta.patient_id, (tiffs instanceof List ? tiffs.size() : 1)] }
        )
        .groupTuple(by: 0)
        .map { pid, counts -> [pid, counts.sum()] }

    ch_all_channels = groupTiffsByPatient(
        ch_deduped_channels
            .combine(ch_channel_counts, by: 0)
            .map { pid, tiff, channels_count -> [pid, channels_count, tiff] }
    )

    // ------------------------------------------------------------------ //
    // 10. EXPORT + PYRAMID (ASSEMBLE_EXPORT, shared with postprocess.nf)
    // ------------------------------------------------------------------ //
    // Given the reused masks, so the rebuilt pyramid carries them when embed_masks is on.
    ASSEMBLE_EXPORT(
        ch_merged_with_qc,
        ch_contours,
        ch_nuc_for_export,
        ch_all_channels,
        ch_masks,
        compartment_mode
    )

    // ------------------------------------------------------------------ //
    // 11. POSTPROCESSING CHECKPOINT — what makes this outdir chainable
    // ------------------------------------------------------------------ //
    // Needed by the next cycle's validateAddCycle. cell_mask is EXTRACT_MASK_SERIES' output,
    // published in this outdir, so the next cycle does not depend on cycle 1's directory.
    POSTPROCESSED_CHECKPOINT(
        ASSEMBLE_EXPORT.out.csv,
        ASSEMBLE_EXPORT.out.geojson,
        ch_merged_with_qc,
        ch_prior_cell_mask,
        ASSEMBLE_EXPORT.out.pyramid
    )

    // ------------------------------------------------------------------ //
    // Versions + size logs
    // ------------------------------------------------------------------ //
    // QUANTIFY_MARKERS / ASSEMBLE_EXPORT versions are already de-duplicated.
    // EXTRACT_MASK_SERIES' versions are not collected.
    ch_versions = Channel.empty()
        .mix(PREPROCESSING.out.versions)
        .mix(ch_adapter_versions)
        .mix(EXTRACT_CELL_PROPERTIES.out.versions.first())
        .mix(SPLIT_CHANNELS.out.versions.first())
        .mix(SPLIT_PRIOR_PYRAMID.out.versions.first())
        .mix(QUANTIFY_MARKERS.out.versions)
        .mix(MERGE_QUANT_CSVS.out.versions.first())
        .mix(NUCLEAR_RETENTION.out.versions.first())
        .mix(CELL_QC.out.versions.first())
        .mix(ASSEMBLE_EXPORT.out.versions)

    ch_size_logs = Channel.empty()
        .mix(SPLIT_CHANNELS.out.size_log)
        .mix(SPLIT_PRIOR_PYRAMID.out.size_log)
        .mix(QUANTIFY_MARKERS.out.size_logs)
        .mix(MERGE_QUANT_CSVS.out.size_log)
        .mix(NUCLEAR_RETENTION.out.size_log)
        .mix(CELL_QC.out.size_log)
        .mix(ASSEMBLE_EXPORT.out.size_logs)
        .mix(EXTRACT_CELL_PROPERTIES.out.size_log)

    if (reg_qc_level >= 1) {
        ch_versions  = ch_versions.mix(GENERATE_REGISTRATION_QC.out.versions.first())
        ch_size_logs = ch_size_logs.mix(GENERATE_REGISTRATION_QC.out.size_log)
    }
    if (do_seg_qc) {
        ch_versions  = ch_versions.mix(ch_seg_qc_versions)
        ch_size_logs = ch_size_logs.mix(ch_seg_qc_size_log)
    }
    if (compartment_mode.compartments) {
        ch_versions  = ch_versions.mix(EXTRACT_NUCLEI_PROPERTIES.out.versions.first())
        ch_size_logs = ch_size_logs.mix(EXTRACT_NUCLEI_PROPERTIES.out.size_log)
    }

    emit:
    geojson     = ASSEMBLE_EXPORT.out.geojson
    merged_csv  = ch_merged_with_qc
    // csv/registered.csv; unused by mirage.nf, emitted for tests.
    checkpoint_csv = REGISTER_PATIENT.out.checkpoint_csv
    // csv/postprocessed.csv, read by the next add_cycle from --prior_outdir.
    postprocessed_checkpoint_csv = POSTPROCESSED_CHECKPOINT.out.csv
    pyramid     = ASSEMBLE_EXPORT.out.pyramid
    qc          = ch_qc
    seg_qc      = ch_seg_qc
    seg_residuals = ch_seg_residuals
    versions    = ch_versions
    size_logs   = ch_size_logs
}
