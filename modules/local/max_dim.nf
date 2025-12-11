nextflow.enable.dsl = 2

process MAX_DIM {
    tag "compute_max_dimensions"
    label 'process_low'

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/metadata", mode: 'copy'

    input:
    path dims_files

    output:
    path "max_dims.txt", emit: max_dims_file

    script:
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
}
