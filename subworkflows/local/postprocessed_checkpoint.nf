/*
========================================================================================
    SUBWORKFLOW: POSTPROCESSED_CHECKPOINT
========================================================================================
    Builds the postprocessing checkpoint rows (Layout.POSTPROCESSED) from the five
    per-patient artifact streams and hands them to CHECKPOINT_WRITER.

    Its own file because add_cycle has no POSTPROCESSING step but must still write this
    manifest: validateAddCycle requires it under --prior_outdir, so without it cycle 3
    could not chain onto cycle 2.

    The columns are owned by lib/Checkpoint.groovy. This file owns the join and which
    Layout kind each artifact is published under.

    Input:
        ch_cell_csv      [meta, file]  per-cell measurement CSV (EXPORT_GEOJSON)
        ch_cell_geojson  [meta, file]  cells.geojson              (EXPORT_GEOJSON)
        ch_merged_csv    [meta, file]  merged quantification table (MERGE_QUANT_CSVS)
        ch_cell_mask     [meta, file]  cell instance mask
        ch_pyramid       [meta, file]  pyramidal OME-TIFF        (MERGE_AND_PYRAMID)

    Output:
        csv: the collected postprocessing checkpoint manifest
========================================================================================
*/

include { CHECKPOINT_WRITER } from './checkpoint_writer'

workflow POSTPROCESSED_CHECKPOINT {
    take:
    ch_cell_csv
    ch_cell_geojson
    ch_merged_csv
    ch_cell_mask
    ch_pyramid

    main:
    // Per-patient join; CHECKPOINT_WRITER collects without blocking other patients.
    ch_base_checkpoint = ch_cell_csv
        .map { meta, csv ->
            def published_path = Layout.publishedPath(params.outdir, meta.patient_id, 'geojson', csv)
            // pixel_size is the same on every stream of a patient; read it from this one.
            [meta.patient_id, published_path, meta.pixel_size]
        }
        .join(ch_cell_geojson.map { meta, geojson ->
            def published_path = Layout.publishedPath(params.outdir, meta.patient_id, 'geojson', geojson)
            [meta.patient_id, published_path]
        })
        .join(ch_merged_csv.map { meta, csv ->
            def published_path = Layout.publishedPath(params.outdir, meta.patient_id, 'quantification', csv)
            [meta.patient_id, published_path]
        })
        .join(ch_cell_mask.map { meta, mask ->
            // publishedOrAsIs: under --start postprocessing or add_cycle the mask is already
            // a published path from an earlier run (publishedPath would double-nest it), so
            // this column can point outside --outdir.
            def published_path = Layout.publishedOrAsIs(params.outdir, meta.patient_id, 'segmentation', mask)
            [meta.patient_id, published_path]
        })
        .join(ch_pyramid.map { meta, pyramid ->
            def published_path = Layout.publishedPath(params.outdir, meta.patient_id, 'pyramid', pyramid)
            [meta.patient_id, published_path]
        })

    ch_checkpoint_rows = ch_base_checkpoint
        .map { patient_id, cell_csv, pixel_size, cell_geojson, merged_csv, cell_mask, pyramid ->
            [
                patient_id  : patient_id,
                // Rows are per patient, so id is the patient id (see lib/Checkpoint.groovy).
                id          : patient_id,
                cell_csv    : cell_csv,
                cell_geojson: cell_geojson,
                merged_csv  : merged_csv,
                cell_mask   : cell_mask,
                pyramid     : pyramid,
                pixel_size  : pixel_size,
            ]
        }

    // Not written at a cleaning level either: cell_mask names an intermediate.
    CHECKPOINT_WRITER(Layout.POSTPROCESSED, ch_checkpoint_rows)
    ch_checkpoint_csv = CHECKPOINT_WRITER.out.csv

    emit:
    csv = ch_checkpoint_csv
}
