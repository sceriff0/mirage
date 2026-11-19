/*
 * PREFLIGHT_SCALE - resolve `--pixel_size` for every input slide from OME metadata alone.
 *
 * Runs once over every slide before any heavy work, reading only OME headers.
 * `--pixel_size auto` with a slide lacking OME PhysicalSizeX/Y fails the run; a supplied
 * number that disagrees with (or cannot be checked against) metadata only warns.
 *
 * The stub runs the real script: it is cheap, and the stub test profile is what
 * exercises the "number supplied, no metadata" warning path.
 */
process PREFLIGHT_SCALE {
    tag "preflight"
    label 'process_single'

    container "bolt3x/mirage-preprocess:1.0.0"

    input:
    path(images, stageAs: 'input_?/*')

    output:
    path("preflight_scale_report.json"), emit: report
    path "versions.yml"                , emit: versions
    path("*.size.csv")                 , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    def pixel_size = params.pixel_size
    """
    # This task fans in over every slide in the run, so one aggregate row, sample id `all`.
    ${ProcessEnvelope.sizeLog(task.process, 'all', ['input_*/*'], 'PREFLIGHT_SCALE.size.csv')}

    # NUL-delimited find, read into an array by hand -- NOT `--images \$(find ...
    # | sort)`: an unquoted command substitution word-splits on every space in a
    # staged filename, so "P001 ref.ome.tiff" became TWO argv elements ("P001"
    # and "ref.ome.tiff") -- both nonexistent, so preflight_scale.py resolved no
    # scale for the real file and reported it missing. `read -r -d ''` is used
    # rather than bash 4's `mapfile -d ''`: this script also runs, unchanged,
    # under the container-less local stub loop (CLAUDE.md), where the shell is
    # whatever `bash` the host provides -- macOS ships bash 3.2, which has no
    # `mapfile` (measured: "mapfile: command not found").
    images=()
    while IFS= read -r -d '' _img; do
        images+=("\$_img")
    done < <(find -L input_* -maxdepth 1 -type f -print0 | sort -z)
    preflight_scale.py \\
        --images "\${images[@]}" \\
        --pixel-size ${pixel_size} \\
        --output preflight_scale_report.json

    ${ProcessEnvelope.versions(task.process, ['tifffile'], task.container)}
    """

    stub:
    def pixel_size = params.pixel_size
    """
    ${ProcessEnvelope.sizeLogStub(task.process, 'all', 'PREFLIGHT_SCALE.size.csv')}

    # NUL-delimited find, read into an array by hand -- NOT `--images \$(find ...
    # | sort)`: an unquoted command substitution word-splits on every space in a
    # staged filename, so "P001 ref.ome.tiff" became TWO argv elements ("P001"
    # and "ref.ome.tiff") -- both nonexistent, so preflight_scale.py resolved no
    # scale for the real file and reported it missing. `read -r -d ''` is used
    # rather than bash 4's `mapfile -d ''`: this script also runs, unchanged,
    # under the container-less local stub loop (CLAUDE.md), where the shell is
    # whatever `bash` the host provides -- macOS ships bash 3.2, which has no
    # `mapfile` (measured: "mapfile: command not found").
    images=()
    while IFS= read -r -d '' _img; do
        images+=("\$_img")
    done < <(find -L input_* -maxdepth 1 -type f -print0 | sort -z)
    preflight_scale.py \\
        --images "\${images[@]}" \\
        --pixel-size ${pixel_size} \\
        --output preflight_scale_report.json

    ${ProcessEnvelope.versionsStub(task.process, ['tifffile'], task.container)}
    """
}
