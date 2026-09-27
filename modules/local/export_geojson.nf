/*
 * EXPORT_GEOJSON - Export cell data to QuPath-compatible GeoJSON
 *
 * Exports all cells with raw marker intensities and morphological measurements
 * in QuPath's native GeoJSON format. No phenotype classification is applied —
 * gating is handled downstream by FlowPath in QuPath.
 *
 * Input:
 *   - Merged quantification CSV with per-cell marker intensities + morphology
 *   - Pre-computed contours JSON for polygon cell boundaries
 * Output:
 *   - GeoJSON with QuPath-native measurement format (array of name/value)
 *   - CSV with raw intensities + z-scores per marker
 */
process EXPORT_GEOJSON {
    tag "${meta.patient_id}"

    container "bolt3x/mirage-quantify:1.0.0"

    input:
    // Stage the nucleus contours under a distinct name: both EXTRACT_CELL_PROPERTIES
    // and EXTRACT_NUCLEI_PROPERTIES emit a file literally named contours.json, and
    // (when compartments are disabled) the same cell-contours file is passed into
    // both slots — either way an unstaged duplicate would collide in the work dir.
    tuple val(meta), path(quant_csv), path(contours_json), path(nucleus_contours_json, stageAs: 'nucleus_contours.json')

    output:
    tuple val(meta), path("export/cells.geojson"), emit: geojson
    // Whole-cell-only companion (no nucleusGeometry), written only in the
    // per-compartment path (params.quantify_compartments). Lighter/faster to import
    // in QuPath; same measurements, so FlowPath compartment gating still works.
    tuple val(meta), path("export/cells_wholecell.geojson"), optional: true, emit: geojson_wholecell
    tuple val(meta), path("export/cells_data.csv"), emit: csv
    path "versions.yml"                            , emit: versions
    path("*.size.csv")                             , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // Per-compartment quantification: pass the nucleus contours (re-keyed to cell
    // labels) so each cell gets a nucleusGeometry in the single combined cells.geojson.
    def nucleus_arg = params.quantify_compartments ? "--nucleus_contours_json ${nucleus_contours_json}" : ''
    // --pixel_size is NOT optional here even though the script has a parameter for it.
    // It was omitted, and export_geojson.py's own argparse default silently supplied
    // 0.325 -- so every "Centroid X µm", "MORPH: Area µm²", "MORPH: Perimeter µm" and axis length in
    // cells.geojson ignored the configured scale entirely. Those measurements are the
    // contract with qupath-extension-flowpath, so the run advertised a scale it was not
    // using. Pass it explicitly; the script now has no default to fall back to.
    //
    // `meta.pixel_size`, NOT `params.pixel_size`: this process is handed a CSV, not an
    // image, so it cannot resolve `params.pixel_size == 'auto'` itself. INPUT_CHECK
    // (subworkflows/local/input_check.nf) already resolved it per-slide via
    // PREFLIGHT_SCALE and carries the number in meta -- see that file's comment. The
    // same substitution is made in conf/modules.config's `withName: 'EXPORT_GEOJSON'`
    // ext.args, which renders a SECOND `--pixel_size` that argparse's last-wins
    // semantics lets override this one; both must stay in sync.
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
