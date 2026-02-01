process GET_IMAGE_DIMS {
    // FIX BUG #5: Use patient_id instead of non-existent meta.id
    tag "${meta.patient_id}"
    label 'process_single'

    container 'docker://bolt3x/attend_image_analysis:preprocess'

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path("*_dims.txt"), emit: dims
    path "versions.yml"                , emit: versions
    path("*.size.csv")                 , emit: size_log

    when:
    task.ext.when == null || task.ext.when

    script:
    // Generate unique prefix using image filename to avoid collisions when multiple images have same channels
    def prefix = task.ext.prefix ?: "${image.simpleName}"
    """
    # Log input size for tracing
    input_bytes=\$(stat --printf="%s" ${image})
    echo "${task.process},${meta.patient_id},${image.name},\${input_bytes}" > ${meta.patient_id}_${image.simpleName}.GET_IMAGE_DIMS.size.csv

    python3 <<'EOF'
from PIL import Image
import sys

# Disable decompression bomb protection for large microscopy images
Image.MAX_IMAGE_PIXELS = None

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
EOF

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        pillow: \$(python3 -c "import PIL; print(PIL.__version__)")
    END_VERSIONS
    """

    stub:
    // Generate unique prefix using image filename to avoid collisions when multiple images have same channels
    def prefix = task.ext.prefix ?: "${image.simpleName}"
    """
    echo "${image.name} 5000 5000" > ${prefix}_dims.txt
    echo "STUB,${meta.patient_id},stub,0" > ${meta.patient_id}_${image.simpleName}.GET_IMAGE_DIMS.size.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        pillow: stub
    END_VERSIONS
    """
}
