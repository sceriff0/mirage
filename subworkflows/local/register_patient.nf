/*
========================================================================================
    SUBWORKFLOW: REGISTER_PATIENT
========================================================================================
    Takes patient-grouped slides — [patient_id, reference_item, all_items] — and
    returns them registered, checkpointed, and with everything the QC stages need.
    The one registration core for both the linear path and add_cycle; grouping
    stays with the callers, which build their groups differently.

    Input:
        ch_grouped: [patient_id, reference_item, all_items]
                    reference_item = [meta, file]
                    all_items      = [[meta, file], ...]  (INCLUDING the reference)
        method:     registration backend name — one of RegBackends.methods(); an
                    unknown name throws in lib/RegBackends.groovy. A plain String
                    argument, not params.registration_method, so add_cycle stays on
                    VALIS (workflows/mirage.nf rejects tiled there).

    Output:
        registered       [meta, file] — registered slides PLUS passthroughs
        images_multi     [meta, file] — the native (pre-registration) slides of
                                        multi-slide patients only; seg-QC input
        checkpoint_csv   the registration checkpoint manifest (see
                         CHECKPOINT_WRITER / Layout)
        transform          [patient_id, registrar.pickle | manifest] — seg-QC warper
        transform_by_slide [meta, manifest] — one per moving slide (empty under VALIS)
        stage_checkpoint   [patient_id, reg_stage_checkpoint/] (VALIS, reg_qc>=2 only)
        intrinsic_tre      the method's own TRE estimate (VALIS CSV / STARE JSON)
        size_logs / versions — the adapter's, unaggregated

    The last six pass through from the adapter unrenamed (see the adapter contract).
========================================================================================
*/

include { VALIS_ADAPTER     } from './adapters/valis_adapter'
include { TILED_ADAPTER     } from './adapters/tiled_adapter'
include { CHECKPOINT_WRITER } from './checkpoint_writer'

workflow REGISTER_PATIENT {
    take:
    ch_grouped   // [patient_id, ref_item, all_items]
    method       // String: one of RegBackends.methods()

    main:
    // Single-slide patients skip the adapter (VALIS crashes on a lone image); the
    // reference passes straight through as the registered output.
    ch_grouped_split = ch_grouped.branch { pid, ref, items ->
        single: items.size() == 1
        multi:  true
    }
    ch_grouped_multi = ch_grouped_split.multi
    // is_passthrough marks an unregistered slide, so the checkpoint does not record it
    // under <pid>/registered/. TILED_ADAPTER sets it on every reference too.
    ch_passthrough   = ch_grouped_split.single.map { _pid, ref, _items ->
        [ref[0] + [is_passthrough: true], ref[1]]
    }  // [meta, file]

    // Per-slide [meta, file] for seg-QC, multi-slide patients only (a lone reference has
    // nothing to score against). seg_qc.nf branches on meta.is_reference.
    ch_images_multi = ch_grouped_multi.flatMap { pid, ref, items -> items }

    // ========================================================================
    // RUN REGISTRATION VIA METHOD-SPECIFIC ADAPTER
    // ========================================================================
    // Every adapter emits the same names, so one binding (ch_adapter) serves both.
    // lib/RegBackends.groovy picks the adapter; a workflow cannot be called by name from a
    // table, so a new adapter also needs an arm here.
    def adapter = RegBackends.of(method).adapter
    if (adapter == 'TILED_ADAPTER') {
        TILED_ADAPTER(ch_grouped_multi)
        ch_adapter = TILED_ADAPTER.out
    } else {
        if (adapter != 'VALIS_ADAPTER')
            error "REGISTER_PATIENT: RegBackends.of('${method}').adapter is '${adapter}', " +
                  "which this dispatch has no arm for. A Nextflow workflow cannot be " +
                  "invoked by name from a table, so a new adapter needs an arm here as " +
                  "well as a row in lib/RegBackends.groovy. Known methods: " +
                  "${RegBackends.methods()}."
        VALIS_ADAPTER(ch_grouped_multi)
        ch_adapter = VALIS_ADAPTER.out
    }

    // Re-introduce single-slide patients (reference passed through unregistered)
    // into the registered stream for QC, checkpointing and postprocessing.
    ch_registered = ch_adapter.registered.mix(ch_passthrough)

    // ========================================================================
    // CHECKPOINT
    // ========================================================================
    // Read by `--start postprocessing` and by add_cycle (from --prior_outdir). Written here
    // so both modes produce it. Columns are owned by lib/Checkpoint.groovy.
    ch_checkpoint_rows = ch_registered
        .map { meta, file ->
            // Where the file will be published; must match the REGISTER / TILED_* publishDir
            // in conf/modules.config. A passthrough slide is recorded where it already is.
            def published_path = meta.is_passthrough
                ? Layout.passthroughPath(params.outdir, meta.patient_id, file, params.skip_preprocessing as boolean)
                : Layout.publishedPath(params.outdir, meta.patient_id, Layout.REGISTERED, file)
            [
                patient_id      : meta.patient_id,
                // From meta, never re-derived from the image basename.
                id              : meta.id,
                registered_image: published_path,
                is_reference    : meta.is_reference,
                channels        : meta.channels.join('|'),
                pixel_size      : meta.pixel_size,
            ]
        }

    CHECKPOINT_WRITER(Layout.REGISTERED, ch_checkpoint_rows)

    emit:
    registered         = ch_registered
    images_multi       = ch_images_multi
    checkpoint_csv     = CHECKPOINT_WRITER.out.csv
    transform          = ch_adapter.transform
    transform_by_slide = ch_adapter.transform_by_slide
    stage_checkpoint   = ch_adapter.stage_checkpoint
    intrinsic_tre      = ch_adapter.intrinsic_tre
    size_logs          = ch_adapter.size_logs
    versions           = ch_adapter.versions
}
