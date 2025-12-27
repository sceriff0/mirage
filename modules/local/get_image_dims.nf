process GET_IMAGE_DIMS {
    // FIX BUG #5: Use patient_id instead of non-existent meta.id
    tag "${meta.patient_id}"
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:preprocess' :
        'docker://bolt3x/attend_image_analysis:preprocess' }"

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path("*_dims.txt"), emit: dims
    path "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    // Generate unique prefix with patient_id and channels
    def prefix = task.ext.prefix ?: "${meta.patient_id}_${meta.channels?.join('_') ?: 'unknown'}"
    """
    #!/usr/bin/env python3
    from PIL import Image
    import sys

    # Open image and get dimensions
    try:
        img = Image.open('${image}')
        width, height = img.size

        # Write dimensions to file (format: filename height width)
        with open('${prefix}_dims.txt', 'w') as f:
            f.write(f"${image.name} {height} {width}\\n")

        print(f"Image dimensions: {height} x {width}")

    except Exception as e:
        print(f"ERROR: Failed to read image: {e}", file=sys.stderr)
        sys.exit(1)

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        pillow: \$(python3 -c "import PIL; print(PIL.__version__)")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.patient_id}_${meta.channels?.join('_') ?: 'unknown'}"
    """
    cat > ${prefix}_dims.txt <<-END
    ${image.name} 5000 5000
    END

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        pillow: stub
    END_VERSIONS
    """
}
