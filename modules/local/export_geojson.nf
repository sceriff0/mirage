/*
 * EXPORT_GEOJSON - export cells to QuPath GeoJSON
 *
 * Every cell with raw marker intensities and morphology as QuPath measurements, plus a
 * CSV with raw intensities and per-marker z-scores. No phenotyping: gating is done in
 * QuPath by FlowPath.
 */
process EXPORT_GEOJSON {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    // Distinct stage name: both contour inputs are named contours.json (and are the
    // same file when compartments are off).
    tuple val(meta), path(quant_csv), path(contours_json), path(nucleus_contours_json, stageAs: 'nucleus_contours.json')

    output:
    tuple val(meta), path("export/cells.geojson"), emit: geojson
    // Whole-cell-only companion (no nucleusGeometry), written only with
    // params.quantify_compartments; lighter to import, same measurements.
    tuple val(meta), path("export/cells_wholecell.geojson"), optional: true, emit: geojson_wholecell
    tuple val(meta), path("export/cells_data.csv"), emit: csv
    path "versions.yml"                            , emit: versions
    path("*.size.csv")                             , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // Nucleus contours (keyed by cell label) give each cell a nucleusGeometry.
    def nucleus_arg = params.quantify_compartments ? "--nucleus_contours_json ${nucleus_contours_json}" : ''
    // Required: every µm measurement in cells.geojson depends on it. meta.pixel_size,
    // not params.pixel_size, because this process has no image to resolve 'auto' from.
    // conf/modules.config's ext.args renders a second --pixel_size that wins (argparse
    // last-wins); keep both in sync.
    """
    ${ProcessEnvelope.sizeLog(task.process, meta.patient_id, ["${quant_csv}"], "${meta.patient_id}.EXPORT_GEOJSON.size.csv")}

    echo "Sample: ${meta.patient_id}"

    mkdir -p export
    export_geojson.py \\
        --cell_data ${quant_csv} \\
        -o export \\
        --pixel_size ${meta.pixel_size} \\
        --contours_json ${contours_json} \\
        ${nucleus_arg} \\
        ${args}

    ${ProcessEnvelope.versions(task.process, ['pandas', 'scipy'], task.container)}
    """

    stub:
    """
    mkdir -p export
    touch export/cells.geojson
    ${params.quantify_compartments ? 'touch export/cells_wholecell.geojson' : ''}
    touch export/cells_data.csv
    ${ProcessEnvelope.sizeLogStub(task.process, meta.patient_id, "${meta.patient_id}.EXPORT_GEOJSON.size.csv")}

    ${ProcessEnvelope.versionsStub(task.process, ['pandas', 'scipy'], task.container)}
    """
}
