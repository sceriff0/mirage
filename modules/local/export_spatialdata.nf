/*
 * EXPORT_SPATIALDATA - serialize postprocessing output as a SpatialData .zarr
 *
 * Cell/nuclei label masks, polygons and an AnnData table (per-compartment intensities
 * + morphology), with registration/segmentation QC in `uns` and per-cell residuals in
 * `obsm`. Nothing is recomputed; this re-serialises POSTPROCESSING's artifacts.
 *
 * The pyramid is included only with --spatialdata_include_image, since it duplicates
 * the largest artifact. Optional inputs are passed as `[]`.
 */
process EXPORT_SPATIALDATA {
    tag "${meta.patient_id}"
    label params.spatialdata_include_image ? 'process_high' : 'process_medium'

    container "bolt3x/mirage-spatialdata:1.0.0"

    input:
    tuple val(meta), path(quant_csv), path(contours_json), path(nucleus_contours_json, stageAs: 'nucleus_contours.json'), path(cell_mask), path(nuclei_mask), path(pyramid)
    path qc_json
    path reg_residuals

    output:
    tuple val(meta), path("spatialdata/${meta.patient_id}.zarr"), emit: zarr
    path "versions.yml", emit: versions
    path("*.size.csv") , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def nuc_mask_arg  = nuclei_mask ? "--nuclei-mask ${nuclei_mask}" : ''
    def nuc_cont_arg  = params.quantify_compartments ? "--nucleus-contours-json ${nucleus_contours_json}" : ''
    def image_arg     = (params.spatialdata_include_image && pyramid) ? "--include-image --pyramid ${pyramid}" : ''
    def qc_arg        = qc_json       ? "--qc-json ${qc_json}"             : ''
    def resid_arg     = reg_residuals ? "--reg-residuals ${reg_residuals}" : ''
    // meta.pixel_size (resolved per slide): without the pyramid the script has no
    // image to resolve params.pixel_size == 'auto' from.
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${quant_csv}"], "${meta.patient_id}.EXPORT_SPATIALDATA.size.csv")}

    mkdir -p spatialdata
    export_spatialdata.py \\
        --quant-csv ${quant_csv} \\
        --contours-json ${contours_json} \\
        --cell-mask ${cell_mask} \\
        ${nuc_mask_arg} \\
        ${nuc_cont_arg} \\
        ${image_arg} \\
        ${qc_arg} \\
        ${resid_arg} \\
        --patient-id ${meta.patient_id} \\
        --pixel-size ${meta.pixel_size} \\
        --residual-join-max-px ${params.spatialdata_residual_join_max_px} \\
        -o spatialdata/${meta.patient_id}.zarr \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['spatialdata', 'anndata', 'geopandas', 'zarr'], task.container)}
    """

    stub:
    """
    mkdir -p spatialdata/${meta.patient_id}.zarr
    echo '{"zarr_format":2}' > spatialdata/${meta.patient_id}.zarr/.zgroup
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}.EXPORT_SPATIALDATA.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['spatialdata', 'anndata', 'geopandas', 'zarr'], task.container)}
    """
}
