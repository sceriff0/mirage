/*
 * SEG_QC_GEOJSON - cell-mask label image -> cell GeoJSON (reg_qc = 2, GeoJSON path)
 *
 * Traces one Polygon per label out of SEG_QC_SEGMENT's `*_cell_mask.tif` and writes a
 * geometry-only cell GeoJSON. WARP_SEG_QC then warps those polygons through the
 * registrar and scores before-vs-after overlap. Segmentation itself is SEGMENT's
 * (aliased in seg_qc.nf), so the QC follows params.seg_method.
 *
 * The output stem must equal the registrar's slide_dict key (VALIS's slide name, which
 * strips .ome.tif/.ome.tiff). Mask filenames differ per backend, so seg_qc.nf resolves
 * the name from the native image and carries it as meta.qc_slide.
 */
process SEG_QC_GEOJSON {
    tag "${meta.patient_id}:${meta.qc_slide}"

    // Same contour tracing as EXTRACT_CELL_PROPERTIES; not a segmentation image, so
    // it does not depend on which backend produced the mask.
    container "bolt3x/mirage-quantify:1.0.0"

    input:
    tuple val(meta), path(cell_mask)

    output:
    tuple val(meta), path("*.geojson"), emit: geojson
    path "versions.yml"               , emit: versions
    path("*.size.csv")                , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    // ext.args carries --tolerance = params.simplify_tolerance, as for EXTRACT_CELL_PROPERTIES.
    def args   = task.ext.args ?: ''
    def prefix = meta.qc_slide
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${cell_mask}"], "${prefix}.SEG_QC_GEOJSON.size.csv")}

    mask_to_geojson.py \\
        --mask ${cell_mask} \\
        --out ${prefix}.geojson \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['skimage', 'scipy'], task.container)}
    """

    stub:
    def prefix = meta.qc_slide
    """
    echo '{"type": "FeatureCollection", "features": []}' > ${prefix}.geojson
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${prefix}.SEG_QC_GEOJSON.size.csv")}
    ${ProcessEnvelope.versionsStub(task.process, ['skimage', 'scipy'], task.container)}
    """
}
