/*
========================================================================================
    SUBWORKFLOW: ASSEMBLE_EXPORT
========================================================================================
    Per-patient export, shared by postprocessing and add_cycle:

      1. EXPORT_GEOJSON  — merged quantification + cell/nucleus contours ->
                           cells.geojson + cells_data.csv
      2. MERGE_AND_PYRAMID — per-patient channel TIFFs -> pyramidal OME-TIFF, with the
                           `embed_masks` gate deciding whether the segmentation masks
                           ride along as a second uint32 OME series.

    Caller differences come in through `take:`, not by branching on params.mode:
      - `ch_nuc_contours`  — real nucleus contours under quantify_compartments, or the
                             cell contours as a harmless placeholder.
      - `ch_pyramid_masks` — [patient_id, cell_mask, nuclei_mask]: freshly segmented in
                             POSTPROCESSING, reused from the prior run in add_cycle.
========================================================================================
*/

include { EXPORT_GEOJSON    } from '../../modules/local/export_geojson'
include { MERGE_AND_PYRAMID } from '../../modules/local/merge_and_pyramid'

workflow ASSEMBLE_EXPORT {
    take:
    ch_merged_csv       // [meta, merged_quant.csv]
    ch_contours         // [patient_id, contours.json]
    ch_nuc_contours     // [patient_id, nucleus_contours.json | contours.json placeholder]
    ch_pyramid_channels // [meta, [per-marker tiffs]]  — one entry per patient
    ch_pyramid_masks    // [patient_id, cell_mask, nuclei_mask]
    compartment_mode    // ParamUtils.compartmentMode(params), resolved in workflows/mirage.nf

    main:

    // ========================================================================
    // EXPORT - QuPath-compatible GeoJSON + raw measurement CSV
    // ========================================================================
    ch_for_export = ch_merged_csv
        .map { meta, csv -> [meta.patient_id, meta, csv] }
        .join(ch_contours, by: 0)
        .join(ch_nuc_contours, by: 0)
        .map { _pid, meta, csv, contours_json, nuc_json ->
            [meta, csv, contours_json, nuc_json]
        }

    EXPORT_GEOJSON(ch_for_export)

    // ========================================================================
    // PYRAMID - merge the per-patient channel TIFFs
    // ========================================================================
    // Optionally embed cell + nuclei masks as a second, single-resolution uint32 OME
    // series (Image:1). Never in the intensity series: a uint32 mask would force the whole
    // image to uint32, which QuPath cannot read as a normal multi-channel image.
    // join (not combine) so a duplicate mask entry errors instead of fanning out.
    def emit_masks = compartment_mode.embedMasks && compartment_mode.compartments && compartment_mode.expanded
    ch_pyramid_in = emit_masks
        ? ch_pyramid_channels
            .map { meta, tiffs -> [meta.patient_id, meta, tiffs] }
            .join(ch_pyramid_masks, by: 0)
            .map { _pid, meta, tiffs, cell_mask, nuclei_mask -> [meta, tiffs, [cell_mask, nuclei_mask]] }
        : ch_pyramid_channels.map { meta, tiffs -> [meta, tiffs, []] }

    // Merge + pyramid in one step, keeping OME-XML metadata (channel names, colours, pixel sizes).
    MERGE_AND_PYRAMID(ch_pyramid_in)

    emit:
    geojson           = EXPORT_GEOJSON.out.geojson
    geojson_wholecell = EXPORT_GEOJSON.out.geojson_wholecell
    csv               = EXPORT_GEOJSON.out.csv
    pyramid           = MERGE_AND_PYRAMID.out.pyramid
    size_logs         = EXPORT_GEOJSON.out.size_log.mix(MERGE_AND_PYRAMID.out.size_log)
    // Versions are de-duplicated here, so callers need not.
    versions          = EXPORT_GEOJSON.out.versions.first()
        .mix(MERGE_AND_PYRAMID.out.versions.first())
}
