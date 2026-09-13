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
    QC surfaces misalignment. It is non-gating in the SCHEDULING sense only --
    every consumer is a collect()/join(), so a missing artifact never deadlocks
    the DAG -- but all three processes below sit under conf/modules.config's
    retry-then-fail selector, so a broken one FAILS THE RUN. See the header above
    that selector; the terminal branch stopped being 'ignore' on 2026-08-25.
      - reg_qc >= 1: DAPI-overlay image QC (GENERATE_REGISTRATION_QC)
      - reg_qc >= 2: + staged seg-overlap QC (SEG_QC_GEOJSON -> WARP_SEG_QC) — per-pair
        IoU and centroid residual at each registration stage, on a correspondence fixed
        after rigid. Uses the classic VALIS registrar pickle from the 2-node graph, plus
        REGISTER's pre-micro stage checkpoint. See docs/registration_qc.md.

    Registration adapter: the classic monolithic VALIS adapter, matching a full run.
    (The distributed/tiled low-memory path was archived 2026-07-24; see git tag
    archive/tiled-valis-2026-07-24.)

    Per-compartment (expanded) quantification is supported: under
    --quantify_compartments, the reused nuclei mask feeds QUANTIFY and
    EXTRACT_NUCLEI_PROPERTIES supplies nucleus contours for EXPORT_GEOJSON.
========================================================================================
*/

include { PREPROCESSING            } from './preprocess'
// The registration adapter dispatch, the single-slide passthrough branch and the
// checkpoint manifest are REGISTER_PATIENT's, shared with registration.nf.
include { REGISTER_PATIENT         } from './register_patient'
include { EXTRACT_MASK_SERIES      } from '../../modules/local/extract_mask_series'
include { EXTRACT_CELL_PROPERTIES  } from '../../modules/local/extract_cell_properties'
include { EXTRACT_NUCLEI_PROPERTIES } from '../../modules/local/extract_nuclei_properties'
include { SPLIT_CHANNELS           } from '../../modules/local/split_channels'
include { SPLIT_CHANNELS as SPLIT_PRIOR_PYRAMID } from '../../modules/local/split_channels'
include { MERGE_QUANT_CSVS         } from '../../modules/local/merge_quant_csvs'
include { GENERATE_REGISTRATION_QC } from '../../modules/local/generate_registration_qc'
include { SEG_QC                   } from './seg_qc'
// Shared with subworkflows/local/postprocess.nf. groupTiffsByPatient is a plain
// function, not a process/workflow, but Nextflow's `include` pulls in either.
include { QUANTIFY_MARKERS; groupTiffsByPatient } from './quantify_markers'
include { ASSEMBLE_EXPORT          } from './assemble_export'
// The postprocessing checkpoint writer, shared with postprocess.nf. Writing it from
// here is what makes an add_cycle --outdir usable as the NEXT cycle's --prior_outdir:
// ParamUtils.validateAddCycle requires both of Layout.ADD_CYCLE_CHECKPOINTS, and
// without this call cycle 3 refused at launch with "required checkpoint
// 'csv/postprocessed.csv' not found". Same repair, one step later, as
// register_patient.nf's CHECKPOINT_WRITER call.
include { POSTPROCESSED_CHECKPOINT } from './postprocessed_checkpoint'

workflow ADD_CYCLE {
    take:
    ch_new_input     // [meta, raw_file]  (new cycle slides, is_reference=false)
    compartment_mode // ParamUtils.compartmentMode(params) — resolved once by
                     // workflows/mirage.nf and threaded down, the same seam
                     // --registration_method has (this file passes 'valis' as a
                     // LITERAL to REGISTER_PATIENT for exactly that reason — see
                     // the REGISTER_PATIENT call below). Only `.compartments` is
                     // read here directly; passed straight through to
                     // ASSEMBLE_EXPORT.

    main:
    // ------------------------------------------------------------------ //
    // 0. PRIOR ASSETS — everything this subworkflow reuses from the prior
    //    completed run, rebuilt from that run's checkpoint CSVs.
    // ------------------------------------------------------------------ //
    // This used to be assembled by the caller (workflows/mirage.nf) and handed
    // over as a bare 7-tuple, which this file then destructured positionally
    // eleven times with `_`-prefixed throwaways. It lives here now, and the
    // per-patient payload is a NAMED map:
    //
    //   [patient_id, [ ref_channels : List<String>  channels of the frozen reference
    //                , ref_image    : path          the frozen registration target
    //                , base_csv     : path          prior merged quantification table
    //                , cell_mask    : path          prior segmentation cell mask
    //                , nuclei_mask  : path          prior segmentation nuclei mask
    //                , pyramid      : path          prior combined pyramid ]]
    //
    // patient_id stays a bare first element so the joins/combines below can key
    // on it with `by: 0`.
    //
    // The prerequisites (--prior_outdir exists, every new-cycle patient appears
    // in the prior postprocessed.csv, --mode add_cycle) are validated in
    // workflows/mirage.nf before this subworkflow is ever invoked.

    // Columns come from lib/Checkpoint.groovy, the writer's owner: this reader
    // never restates the schema. Fail loudly here if the writer's schema drifts
    // from what this reader indexes -- see Checkpoint.requireColumns for why a
    // drift is otherwise silent (splitCsv creates keys only for columns the FILE
    // declares, so a lost column reads as an empty path).
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

    // Columns come from lib/Checkpoint.groovy, the writer's owner: this reader
    // never restates the schema. Only `merged_csv`, `cell_mask` and `pyramid` are
    // used here -- the masks are re-extracted from the pyramid's Image:1 series by
    // EXTRACT_MASK_SERIES below, so the cell_mask column is read and discarded.
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
    // Build [patient_id, ref_item, all_items] with the prior reference marked
    // is_reference=true. The reference is a fixed frame (passed through
    // unregistered by VALIS); we discard its output and keep only the new cycle.
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

    // Registration runs via the classic monolithic VALIS adapter, matching a full run.
    // 'valis' is passed as a LITERAL, not params.registration_method: workflows/mirage.nf
    // rejects --registration_method tiled in add_cycle mode, and going through the shared
    // REGISTER_PATIENT must not be what quietly lifts that rejection. Lifting it is a
    // one-word change here plus deleting the guard — a behaviour change of its own.
    // (The distributed/tiled low-memory VALIS path was archived on 2026-07-24, git tag
    // archive/tiled-valis-2026-07-24; that is a different thing from the STARE backend.)
    //
    // REGISTER_PATIENT also writes csv/registered.csv, from the FULL registered stream —
    // reference row included — exactly as the linear path does. That row is the one a
    // follow-on `--mode add_cycle` reads back (`ch_prior_ref` above filters
    // is_reference=true), so a manifest without it would be one this very file cannot
    // consume. Its single-slide passthrough branch never fires here: every group this
    // file builds is [prior reference, new slide], size 2.
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
    // Per-cell registration residuals (final stage). registration.nf's twin of this
    // block (subworkflows/local/registration.nf:207) has captured SEG_QC.out.per_cell
    // since Group A; this file called SEG_QC without ever capturing it, so add_cycle
    // residuals reached nothing downstream.
    ch_seg_residuals = Channel.empty()
    def reg_qc_level = ParamUtils.regQcLevel(params)

    // Level 2 needs the classic registrar pickle, which the classic VALIS adapter produces.
    def do_seg_qc = reg_qc_level >= 2

    // Level >= 1: before/after DAPI-overlay QC — the new cycle's NATIVE image and its
    // registered counterpart, both against the frozen prior reference.
    //
    // The native is ch_new_pre, which is also what ch_grouped hands to REGISTER_PATIENT
    // above: on this path the only moving slide is the new cycle, and the prior reference
    // is a fixed frame with no "before" of its own. Joined on meta.id for the reason
    // registration.nf's twin of this block gives -- a meta MAP key is one `meta + [k: v]`
    // away from never matching, and join() drops an unmatched pair silently.
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

    // Level >= 2: seg-overlap Dice/IoU. Segment DAPI on the NATIVE (pre-reg)
    // reference + new-cycle images -> cell GeoJSON, then warp through the
    // classic registrar pickle (classic adapter only — see do_seg_qc above).
    // Shares subworkflows/local/seg_qc.nf with registration.nf. 'valis' is passed as a
    // LITERAL for the same reason as the REGISTER_PATIENT call above: SEG_QC takes the method
    // as an ARGUMENT and never reads params.registration_method, so sharing it cannot quietly
    // lift mirage.nf's rejection of --registration_method tiled in this mode.
    // REGISTER_PATIENT.out.transform_by_slide is Channel.empty() under VALIS by the adapters'
    // null-object contract, which is exactly what the valis branch expects.
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
    // INPUT_CHECK already resolved meta.keep_channels, but only ACROSS THIS SHEET.
    // The new-cycle sheet declares no reference (the reference is the prior run's and
    // is never a row here), so the first slide claims every name it carries --
    // including the nuclear channel. Left alone that would be a regression: the
    // priority dedup below is new-wins, so the new cycle's DAPI would REPLACE the
    // frozen reference's DAPI in the rebuilt pyramid.
    //
    // Subtract whatever the prior run's reference already contributes. A re-stained
    // DAPI is therefore dropped as redundant (the behaviour step 9's comment relies
    // on), while a marker the prior run never had -- a cohort that switched nuclear
    // stain mid-study -- now survives instead of being discarded for merely matching
    // params.nuclear_markers.
    //
    // meta.channels_count is NOT adjusted to match: it feeds the deliberate
    // new_count + prior_count over-count at step 9, and over-counting is the safe
    // direction there (the group closes late via remainder:true rather than aborting).
    def prior_ref_channels = CsvUtils.referenceChannelsPerPatient(
        Layout.checkpointCsv(params.prior_outdir, Layout.REGISTERED))

    ch_new_for_split = ch_new_registered.map { meta, f ->
        def alreadyInPrior = ((prior_ref_channels[meta.patient_id] ?: [])
            .collect { it.toString().toUpperCase() }) as Set
        // ABSENT vs EMPTY on the way IN, and never `?:`: INPUT_CHECK always sets
        // meta.keep_channels, and an EMPTY value there is an answer ("this slide's every
        // channel was already claimed within this sheet"), not a missing one. `?:` would
        // treat it as missing and resurrect the slide's whole declared panel.
        def declared = meta.keep_channels != null ? meta.keep_channels : (meta.channels ?: [])
        def keep = declared.findAll { !alreadyInPrior.contains(it.toString().toUpperCase()) }
        // See the Map.plus() note in input_check.nf.
        [meta + [keep_channels: keep], f, false]
    }
    // ... and EMPTY on the way OUT is equally an answer: the subtraction above can empty
    // the keep-set outright (a re-stain that adds nothing the prior run's reference does
    // not already carry). Filter such a slide out for the same reason postprocess.nf does
    // -- `path("*.tiff")` is a mandatory output and a slide with nothing to emit has no
    // work to do. Left in, split_channels.nf rendered NO --keep-channels flag, and the
    // REAL script then fell back to its is_reference=false nuclear rule and re-emitted the
    // prior reference's non-nuclear markers at priority 0, where step 9's new-wins dedup
    // replaced the frozen reference's copies; the stub block's own `?:` diverged further
    // still and emitted meta.channels INCLUDING the nuclear channel. Stub and real
    // disagreed, so no -stub test could have caught it.
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
    // 6. QUANTIFY new markers against the REUSED masks. QUANTIFY reads
    //    params.quantify_compartments to route the nuclei mask; --expanded
    //    arrives via ext.args (conf/modules.config), so no change here.
    //
    //    The per-marker fan-out, the mask combine and the per-patient grouping
    //    are QUANTIFY_MARKERS, shared with postprocess.nf — including the
    //    groupKey(patient_id, channels_count) streaming hint that this file's
    //    former inline copy had lost.
    // ------------------------------------------------------------------ //
    ch_masks = ch_prior_assets.map { pid, prior ->
        [pid, prior.cell_mask, prior.nuclei_mask]
    }
    QUANTIFY_MARKERS(SPLIT_CHANNELS.out.channels, ch_masks)

    // ------------------------------------------------------------------ //
    // 7. MERGE new marker CSVs onto the prior merged table (base) by label
    // ------------------------------------------------------------------ //
    // QUANTIFY_MARKERS emits [meta, csvs]; add_cycle keys the merge by patient
    // and builds its own patient-level meta (no morphology join here — the third
    // MERGE_QUANT_CSVS slot carries the prior base table instead). pixel_size is
    // carried along explicitly: the rebuilt meta below is a fresh literal, not
    // `meta + [...]`, so it would otherwise drop the resolved scale
    // QUANTIFY_MARKERS' input inherited from INPUT_CHECK/PREFLIGHT_SCALE --
    // exactly the cross-cycle scale mismatch POSTPROCESSED_CHECKPOINT (downstream,
    // via ASSEMBLE_EXPORT -> EXPORT_GEOJSON's meta) now requires a real value for.
    ch_new_quant_grouped = QUANTIFY_MARKERS.out.grouped_csv
        .map { meta, csvs -> [meta.patient_id, meta.pixel_size, csvs] }
    ch_base = ch_prior_assets.map { pid, prior -> [pid, prior.base_csv] }
    ch_for_merge = ch_new_quant_grouped
        .combine(ch_base, by: 0)
        .map { pid, pixel_size, csvs, base_csv -> [[patient_id: pid, id: pid, pixel_size: pixel_size], csvs, base_csv] }
    MERGE_QUANT_CSVS(ch_for_merge)

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

    // Deterministic new-wins dedup on a marker-name collision (matches the
    // quantification merge's new-wins rule). Tag new-cycle channels priority 0
    // and prior-pyramid channels priority 1; group by [pid, marker] and keep the
    // lowest priority. An async .mix + first-occurrence .unique would be
    // scheduling-nondeterministic, and could make the pyramid show the OLD image
    // of a re-imaged marker while the quant CSV correctly shows the new one.
    // (Step 5 subtracts the prior reference's channels from the new cycle's keep-set,
    // so any marker the prior pyramid already carries -- DAPI included -- only ever
    // arrives at priority 1 and the frozen reference's copy is preserved. A marker the
    // prior run never had arrives at priority 0 and is unopposed.)
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

    // Per-patient combined channel count, feeding groupTiffsByPatient's
    // channels_count-sized groupKey — the SAME streaming hint postprocess.nf's
    // twin grouping uses (subworkflows/local/quantify_markers.nf, shared by both;
    // this file's own copy used to be a bare `.groupTuple()` with no size hint at
    // all). Deliberately new_count + prior_count WITHOUT subtracting a marker
    // collision: over-counting is the safe direction here (the group still
    // closes complete, just at channel-end via remainder:true rather than exactly
    // on time), whereas under-counting this specific grouping is the one failure
    // mode that ABORTS the run outright — see groupTiffsByPatient's doc comment.
    //
    // Built as `.mix()` of BOTH count streams into a single `groupTuple(by:0)` +
    // `.sum()`, not a `.join()` of two separately-summed streams: a join silently
    // DROPS a patient entirely when either side has no entry for it. SPLIT_CHANNELS
    // and SPLIT_PRIOR_PYRAMID both declare `path("*.tiff")` as a mandatory output,
    // so a missing emission normally means a failed task -- but
    // conf/modules.config's `errorStrategy 'ignore'` branch can swallow exactly
    // that failure, and a join would then turn "pyramid from the surviving side's
    // channels only" into "no pyramid at all, run still green" for that patient.
    // `.mix()` + `groupTuple` + `.sum()` degrades instead of disappearing: a
    // patient present on only one side still gets a count (and downstream, a
    // pyramid) from whatever it does have.
    //
    // Both sides are summed the same way, not just the new-cycle side: a patient
    // can emit more than one entry on EITHER stream. SPLIT_CHANNELS runs once per
    // new-cycle slide when a patient contributes more than one (see ch_grouped's
    // fan-out above), which is why new_count could never be read off a single
    // emission. SPLIT_PRIOR_PYRAMID runs exactly once per patient TODAY (one row
    // per patient in the prior postprocessed checkpoint) -- but summing here costs
    // nothing and removes "the prior checkpoint is single-row-per-patient" as a
    // correctness dependency of this grouping rather than a mere observation about
    // today's writer.
    //
    // One honest caveat: the streaming hint this enables is currently INERT on
    // this path. `ch_deduped_channels`'s own dedup step above is a bare
    // `groupTuple(by:0)` with no size hint (grouping by [pid, marker], not
    // patient_id, so it has no single patient-level count to key on) — an
    // unsized groupTuple cannot know a given key is "done" until the ENTIRE
    // upstream channel closes, so it is already a full barrier. groupTiffsByPatient
    // downstream therefore cannot emit any patient earlier than "every patient's
    // channels have arrived" regardless of how accurately channels_count is sized
    // here. Getting the size right is still correct and still matches
    // postprocess.nf's semantics (see that file's comment for why the two must NOT
    // be assumed numerically equal) — it just does not buy add_cycle any actual
    // parallelism today, unlike postprocess.nf, which has no such upstream barrier.
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
    // ASSEMBLE_EXPORT owns the EXPORT_GEOJSON tuple and the embed_masks gate.
    // ADD_CYCLE feeds it the SAME cell + nuclei masks it reused from
    // ch_prior_assets (they are unchanged here — no re-derivation), so the
    // rebuilt combined pyramid carries the mask series whenever embed_masks is on.
    ASSEMBLE_EXPORT(
        MERGE_QUANT_CSVS.out.merged_csv,
        ch_contours,
        ch_nuc_for_export,
        ch_all_channels,
        ch_masks,
        compartment_mode
    )

    // ------------------------------------------------------------------ //
    // 11. POSTPROCESSING CHECKPOINT — what makes this outdir chainable
    // ------------------------------------------------------------------ //
    // Same writer postprocess.nf uses. Without this call an add_cycle run left
    // only preprocessed.csv + registered.csv behind, so
    // ParamUtils.validateAddCycle refused the NEXT cycle at launch:
    //   "required checkpoint 'csv/postprocessed.csv' not found under
    //    --prior_outdir". Cyclic-IF stopped at two cycles.
    //
    // Nothing is recomputed to satisfy it: every column is an artifact this run
    // already produced. The cell_mask is EXTRACT_MASK_SERIES' output -- the masks
    // re-read out of the prior pyramid's Image:1 series and PUBLISHED to this
    // run's <pid>/segmentation/ (conf/modules.config) -- so the row names a file
    // inside THIS outdir and the next cycle is self-contained rather than
    // transitively depending on cycle 1's directory still existing.
    POSTPROCESSED_CHECKPOINT(
        ASSEMBLE_EXPORT.out.csv,
        ASSEMBLE_EXPORT.out.geojson,
        MERGE_QUANT_CSVS.out.merged_csv,
        ch_prior_cell_mask,
        ASSEMBLE_EXPORT.out.pyramid
    )

    // ------------------------------------------------------------------ //
    // Versions + size logs
    // ------------------------------------------------------------------ //
    // QUANTIFY_MARKERS / ASSEMBLE_EXPORT already applied `.first()` internally
    // (see the comments on their `versions` emits) — do not re-apply it here.
    //
    // EXTRACT_MASK_SERIES.out.versions is deliberately NOT mixed in: it was not
    // collected when the process was invoked from workflows/mirage.nf either, and
    // adding it here would change collated_versions.yml (and therefore the QC
    // report's version table) as a side effect of moving the call. Wiring it up is
    // a real gap, but a behaviour change that belongs in its own commit.
    ch_versions = Channel.empty()
        .mix(PREPROCESSING.out.versions)
        .mix(ch_adapter_versions)
        .mix(EXTRACT_CELL_PROPERTIES.out.versions.first())
        .mix(SPLIT_CHANNELS.out.versions.first())
        .mix(SPLIT_PRIOR_PYRAMID.out.versions.first())
        .mix(QUANTIFY_MARKERS.out.versions)
        .mix(MERGE_QUANT_CSVS.out.versions.first())
        .mix(ASSEMBLE_EXPORT.out.versions)

    ch_size_logs = Channel.empty()
        .mix(SPLIT_CHANNELS.out.size_log)
        .mix(SPLIT_PRIOR_PYRAMID.out.size_log)
        .mix(QUANTIFY_MARKERS.out.size_logs)
        .mix(MERGE_QUANT_CSVS.out.size_log)
        .mix(ASSEMBLE_EXPORT.out.size_logs)
        .mix(EXTRACT_CELL_PROPERTIES.out.size_log)

    if (reg_qc_level >= 1) {
        ch_versions  = ch_versions.mix(GENERATE_REGISTRATION_QC.out.versions.first())
        ch_size_logs = ch_size_logs.mix(GENERATE_REGISTRATION_QC.out.size_log)
    }
    // SEG_QC's emissions already cover SEG_QC_GEOJSON + WARP_SEG_QC (versions pre-.first()).
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
    merged_csv  = MERGE_QUANT_CSVS.out.merged_csv
    // csv/registered.csv. Nothing in workflows/mirage.nf consumes it (collectFile's
    // storeDir writes the file regardless), but it is emitted so a test can assert on
    // it and so the add_cycle path advertises the same artifact REGISTRATION does.
    checkpoint_csv = REGISTER_PATIENT.out.checkpoint_csv
    // csv/postprocessed.csv, on the same terms. This is the manifest a FOLLOW-ON
    // add_cycle reads out of --prior_outdir, so add_cycle now advertises both
    // halves of Layout.ADD_CYCLE_CHECKPOINTS and its outdir can be chained.
    postprocessed_checkpoint_csv = POSTPROCESSED_CHECKPOINT.out.csv
    pyramid     = ASSEMBLE_EXPORT.out.pyramid
    qc          = ch_qc
    seg_qc      = ch_seg_qc
    seg_residuals = ch_seg_residuals
    versions    = ch_versions
    size_logs   = ch_size_logs
}
