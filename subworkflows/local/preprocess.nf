/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { CONVERT_IMAGE           } from '../../modules/local/convert_image'
include { TILE_FOR_BASIC          } from '../../modules/local/tile_for_basic'
include { BASICPY                 } from '../../modules/nf-core/basicpy/main'
include { APPLY_PROFILES          } from '../../modules/local/apply_profiles'
include { GENERATE_PREPROCESS_QC  } from '../../modules/local/generate_preprocess_qc'
include { CHECKPOINT_WRITER       } from './checkpoint_writer'

/*
========================================================================================
    SUBWORKFLOW: PREPROCESSING
========================================================================================
    Description:
        Converts images to standardized OME-TIFF (nuclear/fiducial channel first) and
        applies BaSiC illumination correction: TILE_FOR_BASIC (slide -> tile stack),
        BASICPY (profiles only), APPLY_PROFILES (apply + reassemble).
        params.skip_preprocessing skips the correction, never the conversion.

    Input:
        ch_input: Channel of [meta, file] tuples where meta contains:
                  - patient_id: Patient identifier
                  - is_reference: Boolean indicating reference image
                  - channels: List of channel names (the nuclear/fiducial
                    marker is moved to index 0 by CONVERT_IMAGE)

    Output:
        preprocessed: Preprocessed OME-TIFF images with [meta, file] tuples
        checkpoint_csv: CSV for restart capability
========================================================================================
*/

workflow PREPROCESSING {
    take:
    ch_input  // Channel of [meta, file] tuples

    main:
    // Convert to standardized OME-TIFF format
    CONVERT_IMAGE ( ch_input )

    // Parse output channels from channels.txt file and update metadata
    ch_for_preprocess = CONVERT_IMAGE.out.ome_tiff
        .map { meta, ome_file, channels_file ->
            // Read output channels from file (the nuclear/fiducial channel is first)
            def output_channels = channels_file.text.trim().split(',').toList()

            // Update meta with the output channel order (a new List, not shared).
            def updated_meta = meta + [channels: output_channels]
            [updated_meta, ome_file]
        }

    // BaSiC correction, unless skip_preprocessing. A branch, not ext.when, so the checkpoint
    // names where the file was actually published (<pid>/converted/ vs <pid>/preprocessed/).
    if (params.skip_preprocessing) {
        ch_preprocessed_with_meta = ch_for_preprocess
    } else {
        // nf-core's BASICPY computes profiles only and refuses a single-site image, so
        // TILE_FOR_BASIC puts a tile grid on the Z axis and APPLY_PROFILES applies them.
        TILE_FOR_BASIC ( ch_for_preprocess )

        // The vendored module names outputs from meta.id; it is set for BASICPY's input only.
        BASICPY (
            TILE_FOR_BASIC.out.tiles.map { meta, tiles -> [ meta + [id: tiles.simpleName], tiles ] }
        )

        // Join on "<patient>/<tile basename>", not on meta: BASICPY's meta has the extra id.
        ch_apply = TILE_FOR_BASIC.out.sidecar
            .map { meta, image, json -> [ "${meta.patient_id}/${json.simpleName}", meta, image, json ] }
            .join(
                BASICPY.out.profiles.map { meta, dfp, ffp -> [ "${meta.patient_id}/${meta.id}", dfp, ffp ] }
            )
            .map { key, meta, image, json, dfp, ffp -> [ meta, image, json, dfp, ffp ] }

        APPLY_PROFILES ( ch_apply )
        ch_preprocessed_with_meta = APPLY_PROFILES.out.preprocessed
    }

    // Generate QC images (PNG per channel) for visual inspection
    if (!params.skip_preprocess_qc) {
        GENERATE_PREPROCESS_QC ( ch_preprocessed_with_meta )
    }

    // Checkpoint rows for a restart after preprocessing (written by CHECKPOINT_WRITER).
    ch_checkpoint_rows = ch_preprocessed_with_meta
        .map { meta, image_file ->
            [
                patient_id        : meta.patient_id,
                // From meta, never re-derived from the image basename.
                id                : meta.id,
                // Kinds stay literal arguments: tests/test_layout.py scans for them.
                preprocessed_image: params.skip_preprocessing
                    ? Layout.publishedPath(params.outdir, meta.patient_id, 'converted', image_file)
                    : Layout.publishedPath(params.outdir, meta.patient_id, Layout.PREPROCESSED, image_file),
                is_reference      : meta.is_reference,
                channels          : meta.channels.join('|'),
                pixel_size        : meta.pixel_size,
            ]
        }

    CHECKPOINT_WRITER(Layout.PREPROCESSED, ch_checkpoint_rows)
    ch_checkpoint_csv = CHECKPOINT_WRITER.out.csv

    ch_size_logs = Channel.empty()
        .mix(CONVERT_IMAGE.out.size_log)

    ch_versions = Channel.empty()
        .mix(CONVERT_IMAGE.out.versions.first())

    if (!params.skip_preprocessing) {
        // BASICPY (vendored) emits no size log and its version is a topic value, not a
        // versions.yml; see modules/nf-core/basicpy/MIRAGE-NOTES.md.
        ch_size_logs = ch_size_logs
            .mix(TILE_FOR_BASIC.out.size_log)
            .mix(APPLY_PROFILES.out.size_log)
        ch_versions  = ch_versions
            .mix(TILE_FOR_BASIC.out.versions.first())
            .mix(APPLY_PROFILES.out.versions.first())
    }

    // Collect QC outputs (if enabled)
    ch_preprocess_qc = Channel.empty()
    if (!params.skip_preprocess_qc) {
        ch_preprocess_qc = GENERATE_PREPROCESS_QC.out.qc.map { meta, pngs -> pngs }
        ch_size_logs = ch_size_logs.mix(GENERATE_PREPROCESS_QC.out.size_log)
        ch_versions = ch_versions.mix(GENERATE_PREPROCESS_QC.out.versions.first())
    }

    emit:
    preprocessed   = ch_preprocessed_with_meta
    checkpoint_csv = ch_checkpoint_csv
    preprocess_qc  = ch_preprocess_qc
    size_logs      = ch_size_logs
    versions       = ch_versions
}
