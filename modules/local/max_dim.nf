process MAX_DIM {
    tag "compute_max_dimensions"
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:preprocess' :
        'docker://bolt3x/attend_image_analysis:preprocess' }"

    input:
    path dims_files

    output:
    path "max_dims.txt" , emit: max_dims_file
    path "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    #!/usr/bin/env python3
    import sys
    from pathlib import Path

    # Read all dimension files
    dims_files = sorted(Path('.').glob('*_dims.txt'))

    if not dims_files:
        print("ERROR: No dimension files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(dims_files)} dimension files")

    max_h = 0
    max_w = 0

    for dims_file in dims_files:
        with open(dims_file, 'r') as f:
            line = f.read().strip()
            parts = line.split()
            # Format: filename height width
            h = int(parts[1])
            w = int(parts[2])

            print(f"  {dims_file.name}: {h} x {w}")

            max_h = max(max_h, h)
            max_w = max(max_w, w)

    # Save max dimensions
    with open('max_dims.txt', 'w') as f:
        f.write(f"MAX_HEIGHT {max_h}\\n")
        f.write(f"MAX_WIDTH {max_w}\\n")

    print(f"\\nMaximum dimensions: {max_h} x {max_w}")
    """

    stub:
    """
    cat > max_dims.txt <<-END
    MAX_HEIGHT 10000
    MAX_WIDTH 10000
    END

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
