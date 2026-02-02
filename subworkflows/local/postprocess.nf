nextflow.enable.dsl = 2

/*
========================================================================================
    IMPORT MODULES
========================================================================================
*/

include { SEGMENT               } from '../../modules/local/segment'
include { SPLIT_CHANNELS        } from '../../modules/local/split_channels'
include { QUANTIFY              } from '../../modules/local/quantify'
include { MERGE_QUANT_CSVS      } from '../../modules/local/quantify'
include { PHENOTYPE             } from '../../modules/local/phenotype'
include { MERGE_AND_PYRAMID     } from '../../modules/local/merge_and_pyramid'

/*
========================================================================================
    SUBWORKFLOW:POSTPROCESSING
========================================================================================
    Description:
        Segments reference image, splits multichannel images to single channels,
        quantifies marker intensities per cell, merges results, and assigns phenotypes.

    Input:
        ch_registered: Channel of [meta, file] tuples for registered images

    Output:
        phenotype_csv: Phenotyped cell data CSV
        phenotype_geojson: QuPath-compatible GeoJSON with cell detections
        merged_csv: Merged quantification CSV
        cell_mask: Cell segmentation mask
========================================================================================
*/

workflow POSTPROCESSING {
    take:
    ch_registered       // Channel of [meta, file] tuples

    main:

    // ========================================================================
    // PHENOTYPE CONFIG - Resolve config file (custom or default)
    // ========================================================================
    phenotype_config_ch = params.phenotype_config
        ? Channel.fromPath(params.phenotype_config, checkIfExists: true)
        : Channel.fromPath("${projectDir}/assets/phenotype_config.json", checkIfExists: true)

    // ========================================================================
    // SEGMENTATION - Process reference images only
    // ========================================================================
    ch_references = ch_registered
        .filter { meta, file -> meta.is_reference }

    SEGMENT(ch_references)

    // ========================================================================
    // CHANNEL SPLITTING - Split all multichannel images
    // ========================================================================
    SPLIT_CHANNELS(
        ch_registered.map { meta, file -> [meta, file, meta.is_reference] }
    )

    // ========================================================================
    // QUANTIFICATION - Join channels with their patient's mask
    // ========================================================================
    ch_split_output = SPLIT_CHANNELS.out.channels
        .view { meta, tiffs -> "SPLIT_CHANNELS output: meta.patient_id=${meta.patient_id}, tiffs=${tiffs*.name}" }

    ch_flatmapped = ch_split_output
        .flatMap { meta, tiffs ->
            // Ensure tiffs is always a list (handle both single file and multiple files)
            def tiff_list = tiffs instanceof List ? tiffs : [tiffs]

            // Create unique meta map for each channel file
            tiff_list.collect { tiff ->
                def channel_meta = meta.clone()
                channel_meta.id = "${meta.patient_id}_${tiff.baseName}"
                channel_meta.channel_name = tiff.baseName
                [channel_meta, tiff]
            }
        }
        .view { meta, tiff -> "After flatMap: meta.id=${meta.id}, channel=${meta.channel_name}, tiff=${tiff.name}" }

    ch_for_combine = ch_flatmapped
        .map { meta, tiff -> [meta.patient_id, meta, tiff] }
        .view { patient_id, _meta, _tiff -> "Before combine: key=${patient_id}, channel=${_meta.channel_name}" }

    ch_mask = SEGMENT.out.cell_mask
        .map { meta, mask -> [meta.patient_id, mask] }
        .view { patient_id, _mask -> "Mask available: key=${patient_id}, mask=${_mask.name}" }

    ch_for_quant = ch_for_combine
        .combine(ch_mask, by: 0)
        .view { patient_id, _meta, _tiff, _mask -> "After combine: patient=${patient_id}, channel=${_meta.channel_name}" }
        .map { _patient_id, meta, tiff, mask -> [meta, tiff, mask] }

    QUANTIFY(ch_for_quant)

    // ========================================================================
    // MERGE - Group CSVs by patient_id
    // Deduplicate by patient_id + marker (take first occurrence if same marker appears multiple times)
    // ========================================================================
    ch_grouped_csvs = QUANTIFY.out.individual_csv
        .map { meta, csv ->
            def marker = meta.channel_name  // Extract marker name
            [[meta.patient_id, marker], meta, csv]  // Key by [patient_id, marker]
        }
        .unique { entry -> entry[0] }  // Keep only first occurrence of each [patient_id, marker] pair
        .map { key, meta, csv -> [key[0], meta, csv] }  // Restore to [patient_id, meta, csv]
        .groupTuple(by: 0)
        .map { patient_id, metas, csvs ->
            def meta = metas[0].clone()
            meta.id = patient_id
            [meta, csvs]
        }

    MERGE_QUANT_CSVS(ch_grouped_csvs)

    // ========================================================================
    // CONTOURS - Collect one contours file per patient (all identical since same mask)
    // ========================================================================
    // When export_contours is enabled, each QUANTIFY outputs a contours file
    // We only need one per patient (they're identical). Take first per patient_id.
    ch_contours_raw = params.export_contours
        ? QUANTIFY.out.contours
            .map { meta, contours -> [meta.patient_id, contours] }
            .groupTuple(by: 0)
            .map { patient_id, contours_list -> [patient_id, contours_list[0]] }  // Take first (all identical)
        : Channel.empty()

    // Join merged CSV with contours (left join - use placeholder if no contours)
    ch_merged_with_contours = MERGE_QUANT_CSVS.out.merged_csv
        .map { meta, csv -> [meta.patient_id, meta, csv] }
        .join(ch_contours_raw, by: 0, remainder: true)
        .map { patient_id, meta, csv, contours ->
            def contours_file = contours ?: file('NO_CONTOURS')
            [meta, csv, contours_file]
        }

    // ========================================================================
    // PHENOTYPING - Run on merged CSV with configurable rules
    // ========================================================================
    PHENOTYPE(
        ch_merged_with_contours,  // [meta, csv, contours_file]
        phenotype_config_ch.first()  // .first() converts to value channel for reuse across samples
    )

    // ========================================================================
    // MERGE - Combine split channel TIFFs with segmentation mask (per patient)
    // ========================================================================
    // Group split channel TIFFs by patient for merging
    // SPLIT_CHANNELS already handles DAPI filtering correctly
    // Deduplicate by patient_id + marker to avoid duplicate channel names
    ch_split_grouped = SPLIT_CHANNELS.out.channels
        .flatMap { meta, tiffs ->
            // Normalize to List and create entries keyed by [patient_id, marker]
            def tiff_list = tiffs instanceof List ? tiffs : [tiffs]
            tiff_list.collect { tiff ->
                [meta.patient_id, tiff.baseName, tiff]
            }
        }
        .unique { patient_id, marker, _tiff -> [patient_id, marker] }  // Keep first occurrence of each patient+marker
        .map { patient_id, _marker, tiff -> [patient_id, tiff] }
        .groupTuple(by: 0)
        .map { patient_id, tiffs ->
            // Create patient-level metadata
            def patient_meta = [
                patient_id: patient_id,
                is_reference: false  // Not relevant at patient level
            ]
            [patient_meta, tiffs]
        }

    // Join split channels with segmentation mask for MERGE
    ch_for_merge = ch_split_grouped
        .map { meta, tiffs -> [meta.patient_id, meta, tiffs] }
        .join(
            SEGMENT.out.cell_mask.map { meta, mask -> [meta.patient_id, mask] },
            by: 0
        )
        .map { _patient_id, meta, split_tiffs, cell_mask ->
            [meta, split_tiffs, cell_mask]
        }

    // MERGE_AND_PYRAMID combines merge + pyramid generation in one step
    // This preserves OME-XML metadata (channel names, colors, pixel sizes)
    // and generates QuPath-compatible pyramidal OME-TIFF directly
    MERGE_AND_PYRAMID(ch_for_merge)

    // ========================================================================
    // CHECKPOINT - Collect all outputs by patient
    // ========================================================================
    // Use collectFile() for non-blocking aggregation (enables patient-level parallelism)
    // The join chain is kept (it's per-patient and doesn't block other patients)
    ch_checkpoint_csv = PHENOTYPE.out.csv
        .map { meta, csv ->
            def published_path = "${params.outdir}/${meta.patient_id}/phenotype/${csv.name}"
            [meta.patient_id, published_path]
        }
        .join(PHENOTYPE.out.geojson.map { meta, geojson ->
            def published_path = "${params.outdir}/${meta.patient_id}/phenotype/${geojson.name}"
            [meta.patient_id, published_path]
        })
        .join(PHENOTYPE.out.mapping.map { meta, map ->
            def published_path = "${params.outdir}/${meta.patient_id}/phenotype/${map.name}"
            [meta.patient_id, published_path]
        })
        .join(MERGE_QUANT_CSVS.out.merged_csv.map { meta, csv ->
            def published_path = "${params.outdir}/${meta.patient_id}/quantification/${csv.name}"
            [meta.patient_id, published_path]
        })
        .join(SEGMENT.out.cell_mask.map { meta, mask ->
            def published_path = "${params.outdir}/${meta.patient_id}/segmentation/${mask.name}"
            [meta.patient_id, published_path]
        })
        .join(MERGE_AND_PYRAMID.out.pyramid.map { meta, pyramid ->
            def published_path = "${params.outdir}/${meta.patient_id}/pyramid/${pyramid.name}"
            [meta.patient_id, published_path]
        })
        .map { patient_id, pheno_csv, pheno_geojson, pheno_map, merged_csv, cell_mask, pyramid ->
            "${patient_id},${pheno_csv},${pheno_geojson},${pheno_map},${merged_csv},${cell_mask},${pyramid}"
        }
        .collectFile(
            name: 'postprocessed.csv',
            newLine: true,
            storeDir: "${params.outdir}/csv",
            seed: 'patient_id,phenotype_csv,phenotype_geojson,phenotype_mapping,merged_csv,cell_mask,pyramid'
        )

    // Collect size logs from all postprocessing processes
    ch_size_logs = Channel.empty()
        .mix(SEGMENT.out.size_log)
        .mix(SPLIT_CHANNELS.out.size_log)
        .mix(QUANTIFY.out.size_log)
        .mix(MERGE_QUANT_CSVS.out.size_log)
        .mix(PHENOTYPE.out.size_log)
        .mix(MERGE_AND_PYRAMID.out.size_log)

    emit:
    checkpoint_csv = ch_checkpoint_csv
    size_logs = ch_size_logs
}
