/*
========================================================================================
    SUBWORKFLOW: REGISTER_PATIENT
========================================================================================
    Takes patient-grouped slides — [patient_id, reference_item, all_items] — and
    returns them registered, checkpointed, and with everything the QC stages need.

    WHY THIS EXISTS. There were two implementations of this. The linear path went
    through REGISTRATION; subworkflows/local/add_cycle.nf called VALIS_ADAPTER
    directly and re-did the surrounding wiring by hand. The copy had already lost,
    by construction, three things the original has:

      - the checkpoint manifest (an add_cycle run wrote none, so its --outdir could
        never be a second add_cycle's --prior_outdir),
      - the single-slide passthrough branch, and
      - the tiled/STARE backend.

    This file is now the only place that answers "how does a patient group become a
    registered stream?". Both callers assemble the group their own way — the linear
    path groups a preprocessed stream with a sized groupKey; add_cycle pairs each
    new slide with the FROZEN prior reference read out of --prior_outdir — and
    that difference is real, so grouping
    deliberately stays with the callers.

    Input:
        ch_grouped: [patient_id, reference_item, all_items]
                    reference_item = [meta, file]
                    all_items      = [[meta, file], ...]  (INCLUDING the reference)
        method:     registration backend name — one of RegBackends.methods().
                    lib/RegBackends.groovy maps it to an adapter; an unknown name
                    throws there rather than falling through to VALIS.

                    A plain String, not a channel: Nextflow binds workflow `take:`
                    values verbatim (same pattern as INPUT_CHECK's image_column).
                    It is an ARGUMENT rather than a read of
                    params.registration_method so that add_cycle keeps its current,
                    hard-wired VALIS behaviour exactly — workflows/mirage.nf rejects
                    --registration_method tiled in add_cycle mode, and until that
                    rejection is deliberately lifted this subworkflow must not be the
                    thing that quietly lifts it.

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

    Those last six are passed through from the adapter UNRENAMED. Both adapters emit the
    same vocabulary and fill any slot their method lacks with `Channel.empty()` (the
    contract is written out in either adapter's header), so this file holds no
    translation table and nothing below it has to know which backend ran.
========================================================================================
*/

include { VALIS_ADAPTER     } from './adapters/valis_adapter'
include { TILED_ADAPTER     } from './adapters/tiled_adapter'
include { CHECKPOINT_WRITER } from './checkpoint_writer'

workflow REGISTER_PATIENT {
    take:
    ch_grouped   // [patient_id, ref_item, all_items]
    method       // String: one of RegBackends.methods() — see the Input doc above

    main:
    // Single-slide patients (only the reference, nothing to register) must NOT
    // be sent to the registration adapter: VALIS crashes on a lone image
    // ("negative dimensions are not allowed" / "M is None — no transformation
    // matrix"). For such a patient the reference IS the registered output, so we
    // branch it out here and pass it straight through to ch_registered below.
    ch_grouped_split = ch_grouped.branch { pid, ref, items ->
        single: items.size() == 1
        multi:  true
    }
    ch_grouped_multi = ch_grouped_split.multi
    // is_passthrough marks a slide that reaches the registered stream WITHOUT having
    // been registered. It is what the checkpoint writer keys on: nothing publishes an
    // unregistered slide into <pid>/registered/, so recording it there names a file
    // that does not exist (tests/checkpoint_manifest.nf.test).
    // TILED_ADAPTER sets the same flag on every patient's reference, which likewise
    // passes through unwarped.
    ch_passthrough   = ch_grouped_split.single.map { _pid, ref, _items ->
        [ref[0] + [is_passthrough: true], ref[1]]
    }  // [meta, file]

    // Flattened back to [meta, file] for consumers (seg-QC) that segment individual slides
    // rather than patient groups. Single-slide patients are excluded here on purpose: they
    // have no moving slide to score against, so segmenting their reference would compute a
    // full GPU StarDist WSI segmentation and discard it.
    //
    // NOTE: unlike the raw ch_preprocessed stream, `items` here comes out of the caller's
    // grouping closure. Every meta reaching it carries the is_reference the SAMPLESHEET
    // declared -- nothing is filled in, since auto-promotion was removed and a patient
    // with no declared reference is now rejected upstream. seg_qc.nf's reference/moving
    // branch reads meta.is_reference, so this
    // channel is what lets an auto-reference multi-slide patient be scored at all — sourced
    // from the raw stream, that patient's reference branch was empty and it silently
    // produced zero seg-QC output.
    ch_images_multi = ch_grouped_multi.flatMap { pid, ref, items -> items }

    // ========================================================================
    // RUN REGISTRATION VIA METHOD-SPECIFIC ADAPTER
    // ========================================================================
    // The ENTIRE method dispatch. Every adapter emits the identical vocabulary
    // (registered / transform / transform_by_slide / stage_checkpoint / intrinsic_tre /
    // size_logs / versions), with Channel.empty() in any slot a method cannot fill — see
    // any adapter's header for the contract. So this binds ONE name, `ch_adapter`,
    // instead of a per-emit translation table plus the pre-declared empty channels that
    // used to exist only so the union of the vocabularies could be assembled here.
    //
    // WHICH ADAPTER IS lib/RegBackends.groovy's ANSWER, NOT THIS FILE'S. It used to be a
    // three-arm if/else over method-name literals, which was itself a repair of an
    // earlier `if (tiled) ... else VALIS` that registered any unknown method with VALIS
    // and reported success. RegBackends.of() throws on a name the table does not know,
    // so the unknown-METHOD case is closed before this branch runs; the else-arm's own
    // check below closes the other half — a method the table knows whose adapter this
    // dispatch has no arm for, which is what a third backend would be on the day it is
    // added to the table and not here.
    //
    // A two-arm `if` and not a lookup: a Nextflow workflow cannot be invoked by name out
    // of a map, so the CALL has to be written out. Only the DECISION moved.
    //
    // NOTE: the *old distributed-VALIS* low-memory path was archived 2026-07-24
    // (git tag archive/tiled-valis-2026-07-24). The 'tiled' backend is a SEPARATE,
    // live STARE backend — don't confuse the two.
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
    // The manifest is not a nicety of the linear path: it is the file
    // `--start postprocessing` reads, and the file `mode='add_cycle'` reads out of
    // `--prior_outdir` to recover the frozen reference. It used to be owned by
    // subworkflows/local/registration.nf alone, so an add_cycle run -- which never goes
    // through REGISTRATION -- wrote none at all, and its `--outdir` could therefore
    // never be a second add_cycle's `--prior_outdir`. Writing it HERE, in the one
    // registration core both modes go through, is what closes that; it briefly lived in
    // a file of its own for the same reason, which stopped being necessary once the
    // WRITE itself moved to CHECKPOINT_WRITER.
    //
    // The row format is the contract with every reader (add_cycle.nf's `ch_prior_ref`,
    // CsvUtils' checkpoint validation, the `--start` samplesheet parser). It is owned by
    // lib/Checkpoint.groovy -- the columns are named nowhere here, they are asked for.
    ch_checkpoint_rows = ch_registered
        .map { meta, file ->
            // Where the file WILL be published. This must agree with REGISTER's /
            // TILED_*'s publishDir in conf/modules.config, including the producer
            // subdirectory those blocks' `pattern:` carries along ('registered_slides/'
            // for VALIS, 'registered/' for tiled). Both rules live in Layout.
            //
            // A passthrough slide was never registered, so no registration process
            // published it and <pid>/registered/ may not even exist; Layout.passthroughPath
            // records where it actually is instead.
            def published_path = meta.is_passthrough
                ? Layout.passthroughPath(params.outdir, meta.patient_id, file, params.skip_preprocessing as boolean)
                : Layout.publishedPath(params.outdir, meta.patient_id, Layout.REGISTERED, file)
            [
                patient_id      : meta.patient_id,
                // RULING R17: carried forward from meta, never re-derived from
                // registered_image's basename below -- see lib/Checkpoint.groovy.
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
    // Straight through from the adapter, under the adapter's own names — see the header.
    transform          = ch_adapter.transform
    transform_by_slide = ch_adapter.transform_by_slide
    stage_checkpoint   = ch_adapter.stage_checkpoint
    intrinsic_tre      = ch_adapter.intrinsic_tre
    size_logs          = ch_adapter.size_logs
    versions           = ch_adapter.versions
}
