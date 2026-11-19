/*
========================================================================================
    SUBWORKFLOW: SEG_QC  (reg_qc = 2 segmentation-overlap QC)
========================================================================================
    Shared by REGISTRATION and ADD_CYCLE.

    Segments each slide on its native (pre-registration) image with the run's own
    segmenter (SEG_QC_SEGMENT is SEGMENT under an alias, so params.seg_method applies),
    traces the cell mask into GeoJSON, warps the polygons through the method's transform
    and scores before-vs-after overlap (per-pair IoU, dice_matched) plus per-cell centroid
    residuals. Segmenting the native image keeps interpolation out of the score.

    The scorer is method-agnostic; only the warper differs:
      * valis  -> per-PATIENT transform, BioFormats JVM, pre-micro checkpoint
      * tiled  -> per-SLIDE transform, JVM-free, no checkpoint
    WARP_SEG_QC's per-method settings come from lib/WarpBackends.groovy; this file only
    shapes the input tuple.

    `method` is an argument, never params.registration_method (add_cycle passes 'valis').
    The transform channel the taken branch does not read arrives as `Channel.empty()`.

    Input:
        ch_native_images      [meta, file]            slides to segment on their native grid
        ch_transform          [patient_id, transform] per-PATIENT transform (valis branch)
        ch_stage_checkpoint   [patient_id, ckpt_dir]  REGISTER pre-micro checkpoint, may be empty
        ch_transform_by_slide [meta, transform]       per-MOVING-SLIDE transform (tiled branch)
        method                String                  one of RegBackends.methods(); the join
                                                        shape comes from RegBackends.segQcJoin

    Output:
        metrics   [meta, *_seg_qc.json]
        per_cell  [meta, *_reg_residuals.csv]  (optional output of the warp process)
        size_log  size logs of SEG_QC_SEGMENT + SEG_QC_GEOJSON + whichever warp process ran
        versions  versions.yml of SEG_QC_SEGMENT + SEG_QC_GEOJSON + whichever warp process
                  ran (already .first())
========================================================================================
*/

include { SEGMENT as SEG_QC_SEGMENT } from '../../modules/local/segment'
include { SEG_QC_GEOJSON            } from '../../modules/local/seg_qc_geojson'
include { WARP_SEG_QC               } from '../../modules/local/warp_seg_qc'

workflow SEG_QC {
    take:
    ch_native_images
    ch_transform
    ch_stage_checkpoint
    ch_transform_by_slide
    method

    main:
    // Segment the native slides with SEGMENT, aliased. The alias inherits SEGMENT's
    // withName: config, including publishDir; the later `withName: 'SEG_QC_SEGMENT'` block
    // turns publishing off so these masks stay out of <pid>/segmentation/
    // (tests/test_process_alias_config.py).
    def seg_params = SegBackends.ctxParams(params)

    // Slide name from the native image, carried as meta.qc_slide. It must equal VALIS's
    // slide_dict key (extension stripped), and mask filenames differ per backend, so it is
    // computed here rather than recovered downstream.
    def slide_of = { img -> img.name.replaceAll(/\.ome\.tiff?$/, '').replaceAll(/\.tiff?$/, '') }
    def ch_geojson
    def ch_seg_size = Channel.empty()
    def ch_seg_versions = Channel.empty()
    if (params.seg_qc_nuclei_dir) {
        // Reuse another run's <pid>/qc/registration/geojson/<qc_slide>.geojson. Valid only if
        // it segmented the same native slides with the same seg_method; a missing slide errors.
        def src = params.seg_qc_nuclei_dir.toString()
        ch_geojson = ch_native_images.map { meta, img ->
            def slide = slide_of.call(img)
            def gj = file("${src}/${meta.patient_id}/qc/registration/geojson/${slide}.geojson")
            if (!gj.exists())
                error "--seg_qc_nuclei_dir: no QC nuclei for slide '${slide}' of patient " +
                      "${meta.patient_id} at ${gj}. That run must have segmented the same " +
                      "native slides (reg_qc=2) with the same seg_method."
            tuple(meta + [qc_slide: slide], gj)
        }
    } else {
        ch_to_segment = ch_native_images.map { meta, img ->
            tuple(meta + [qc_slide: slide_of.call(img)], img, seg_params)
        }
        SEG_QC_SEGMENT(ch_to_segment)

        // Label image -> polygons; every backend emits the same `*_cell_mask.tif`.
        SEG_QC_GEOJSON(SEG_QC_SEGMENT.out.cell_mask)
        ch_geojson = SEG_QC_GEOJSON.out.geojson
        ch_seg_size = SEG_QC_SEGMENT.out.size_log.mix(SEG_QC_GEOJSON.out.size_log)
        ch_seg_versions = SEG_QC_SEGMENT.out.versions.first()
            .mix(SEG_QC_GEOJSON.out.versions.first())
    }

    ch_gj = ch_geojson.branch { meta, gj ->
        reference: meta.is_reference
        moving:    !meta.is_reference
    }
    // reference: [patient_id, ref_geojson, ref_slide_name(=stem)]
    ch_ref_gj = ch_gj.reference.map { meta, gj -> [meta.patient_id, gj, gj.simpleName] }
    // moving: [patient_id, meta, moving_geojson, moving_slide_name]
    ch_mov_gj = ch_gj.moving.map { meta, gj -> [meta.patient_id, meta, gj, gj.simpleName] }

    // Both arms build WARP_SEG_QC's tuple [meta, method, transform, ref_slide, moving_slide,
    // ref_geojson, moving_geojson, stage_checkpoint]; they differ in how transform and
    // checkpoint are joined. The shape comes from RegBackends.segQcJoin, not the method name.
    if (RegBackends.segQcJoin(method) == 'per_slide') {
        // Tiled: one transform per moving slide, joined on (patient, sorted channels).
        // No stage checkpoint, so `[]`.
        ch_manifest_keyed = ch_transform_by_slide
            .map { meta, m -> [meta.patient_id, meta.channels.toSorted().join('|'), m] }

        ch_for_warp = ch_mov_gj
            .map { pid, meta, gj, name -> [pid, meta.channels.toSorted().join('|'), meta, gj, name] }
            .combine(ch_ref_gj, by: 0)
            .combine(ch_manifest_keyed, by: [0, 1])
            .map { _pid, _sig, meta, mov_gj, mov_name, ref_gj, ref_name, m ->
                tuple(meta, method, m, ref_name, mov_name, ref_gj, mov_gj, [])
            }
    } else {
        // VALIS: one registrar per patient. One checkpoint entry per patient (`[]` if none):
        // a plain combine on the optional channel would drop patients that lack it.
        ch_ckpt_by_patient = ch_transform
            .map { pid, _pickle -> tuple(pid, []) }
            .join(ch_stage_checkpoint, by: 0, remainder: true)
            .map { pid, _placeholder, ckpt -> tuple(pid, ckpt ?: []) }

        ch_for_warp = ch_mov_gj
            .combine(ch_ref_gj, by: 0)
            .combine(ch_transform, by: 0)
            .combine(ch_ckpt_by_patient, by: 0)
            .map { pid, meta, mov_gj, mov_name, ref_gj, ref_name, pickle, ckpt ->
                tuple(meta, method, pickle, ref_name, mov_name, ref_gj, mov_gj, ckpt)
            }
    }

    WARP_SEG_QC(ch_for_warp)
    ch_metrics       = WARP_SEG_QC.out.metrics
    ch_per_cell      = WARP_SEG_QC.out.per_cell
    ch_warp_size     = WARP_SEG_QC.out.size_log
    ch_warp_versions = WARP_SEG_QC.out.versions

    emit:
    metrics  = ch_metrics
    per_cell = ch_per_cell
    size_log = ch_seg_size.mix(ch_warp_size)
    versions = ch_seg_versions.mix(ch_warp_versions.first())
}
