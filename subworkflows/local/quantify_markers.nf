/*
========================================================================================
    SUBWORKFLOW: QUANTIFY_MARKERS
========================================================================================
    The per-marker quantification chain, shared by postprocessing and add_cycle:

        split-channel stacks -> per-marker fan-out -> QUANTIFY -> per-patient group

    The grouping carries the `groupKey(patient_id, channels_count)` streaming hint.
    Callers differ only in where the masks come from and what they do with the
    grouped CSVs; nothing here branches on params.mode.

    Also exports `groupTiffsByPatient`, the pyramid-channel grouping both callers use.
========================================================================================
*/

include { QUANTIFY } from '../../modules/local/quantify'

// The pipeline's only --debug_channels view helper. Off unless --debug_channels, log-only.
def viewIfDebug(channel, Closure formatter) {
    return params.debug_channels ? channel.view(formatter) : channel
}

// Groups one-tiff-per-marker entries into a per-patient list for ASSEMBLE_EXPORT, with a
// channels_count-sized groupKey. An under-count splits the group, and a one-file group
// aborts in MERGE_AND_PYRAMID's memory closure.
//
// ch_tagged: [patient_id, channels_count, tiff], already deduplicated by [patient_id, marker]
// by the caller. A null channels_count falls back to a bare (non-streaming) key.
def groupTiffsByPatient(ch_tagged) {
    return ch_tagged
        .map { patient_id, channels_count, tiff ->
            def gkey = channels_count
                ? groupKey(patient_id, channels_count)
                : patient_id
            [gkey, tiff]
        }
        .groupTuple(by: 0, remainder: true)
        .map { patient_id, tiffs_unordered ->
            // Sort: groupTuple emits in arrival order and path lists hash positionally.
            // One list only, so there is no pairing to break.
            def tiffs = tiffs_unordered.toSorted { it.name }
            // Extract actual patient_id from groupKey wrapper if needed
            def pid = patient_id.toString()
            def patient_meta = [
                id: pid,
                patient_id: pid,
                is_reference: false  // Not relevant at patient level
            ]
            [patient_meta, tiffs]
        }
}

workflow QUANTIFY_MARKERS {
    take:
    ch_split_channels   // [meta, tiffs]                        — SPLIT_CHANNELS.out.channels
    ch_masks            // [patient_id, cell_mask, nuclei_mask]

    main:

    ch_split_viewed = viewIfDebug(
        ch_split_channels,
        { meta, tiffs -> "SPLIT_CHANNELS output: patient=${meta.patient_id}, tiffs=${tiffs*.name}" }
    )

    ch_flatmapped = ch_split_viewed
        .flatMap { meta, tiffs ->
            // Ensure tiffs is always a list (handle both single file and multiple files)
            def tiff_list = tiffs instanceof List ? tiffs : [tiffs]

            // One meta per channel file. meta.channels stays a shared List: read it with
            // toSorted(), never sort(), which would mutate every copy.
            tiff_list.collect { tiff ->
                def channel_meta = meta + [
                    id: "${meta.patient_id}_${tiff.baseName}",
                    channel_name: tiff.baseName
                ]
                [channel_meta, tiff]
            }
        }
    ch_flatmapped = viewIfDebug(
        ch_flatmapped,
        { meta, tiff -> "After flatMap: id=${meta.id}, channel=${meta.channel_name}, tiff=${tiff.name}" }
    )

    ch_for_combine = ch_flatmapped
        .map { meta, tiff -> [meta.patient_id, meta, tiff] }
    ch_for_combine = viewIfDebug(
        ch_for_combine,
        { patient_id, _meta, _tiff -> "Before combine: key=${patient_id}, channel=${_meta.channel_name}" }
    )

    // Masks keyed by patient_id; QUANTIFY reads the nuclear one only under quantify_compartments.
    ch_masks_viewed = viewIfDebug(
        ch_masks,
        { patient_id, _cell, _nuc -> "Masks available: key=${patient_id}, cell=${_cell.name}, nuclei=${_nuc.name}" }
    )

    // combine(by: 0), not join: one mask pair fans out over the patient's N markers.
    ch_for_quant = ch_for_combine
        .combine(ch_masks_viewed, by: 0)
        .map { _patient_id, meta, tiff, cell_mask, nuclei_mask -> [meta, tiff, cell_mask, nuclei_mask] }
    ch_for_quant = viewIfDebug(
        ch_for_quant,
        { meta, _tiff, _cell, _nuc -> "After combine: patient=${meta.patient_id}, channel=${meta.channel_name}" }
    )

    QUANTIFY(ch_for_quant)

    // ========================================================================
    // GROUP - per-marker CSVs by patient, deduplicated on [patient_id, marker]
    // ========================================================================
    // groupKey(channels_count) streams each patient as soon as its markers arrive.
    // channels_count is exact: it comes from the same resolver that picks the markers
    // SPLIT_CHANNELS emits. remainder: true is a safety net. An over-count only emits late;
    // an under-count emits the patient twice, which makes add_cycle run MERGE_QUANT_CSVS
    // twice and makes postprocess.nf's join silently drop markers.
    ch_grouped_csvs = QUANTIFY.out.individual_csv
        .map { meta, csv ->
            def marker = meta.channel_name  // Extract marker name
            [[meta.patient_id, marker], meta, csv]  // Key by [patient_id, marker]
        }
        .unique { entry -> entry[0] }  // Keep only first occurrence of each [patient_id, marker] pair
        .map { key, meta, csv ->
            // Use groupKey for streaming if channels_count is available
            def gkey = meta.channels_count
                ? groupKey(key[0], meta.channels_count)
                : key[0]
            [gkey, meta, csv]
        }
        .groupTuple(by: 0, remainder: true)
        .map { patient_id, metas_unordered, csvs_unordered ->
            // groupTuple emits in arrival order, so sort for stable -resume hashes and a
            // stable metas[0]. Pair first, then sort; groupTuple(sort:) would re-pair wrongly.
            def paired = [metas_unordered, csvs_unordered].transpose()
                            .toSorted { it[0].channel_name }
            def metas = paired.collect { it[0] }
            def csvs  = paired.collect { it[1] }
            // patient_id.toString() unwraps the groupKey.
            def meta = metas[0] + [id: patient_id.toString()]
            [meta, csvs]
        }

    emit:
    // [meta, [per-marker quant csvs]] — one entry per patient.
    grouped_csv = ch_grouped_csvs
    size_logs   = QUANTIFY.out.size_log
    // Versions are de-duplicated here, so callers need not.
    versions    = QUANTIFY.out.versions.first()
}
