nextflow.enable.dsl = 2

process GPU_REGISTER {
    tag "${moving.simpleName}"
    label 'gpu'
    container "${params.container.register_gpu}"

    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered", mode: 'copy', pattern: "*.ome.tiff"
    publishDir "${params.outdir}/${params.id}/${params.registration_method}/registered_qc", mode: 'copy', pattern: "qc/*"

    input:
    tuple path(reference), path(moving)

    output:
    path "${moving.simpleName}_registered.ome.tiff", emit: registered
    path "qc/*_QC_RGB.tif"                         , emit: qc, optional: true

    script:
    def crop_size = params.gpu_reg_crop_size ?: 2000
    def overlap = params.gpu_reg_overlap ?: 200
    def n_features = params.gpu_reg_n_features ?: 2000
    def n_workers = params.gpu_reg_n_workers ?: 4
    """
    mkdir -p qc

    register_gpu.py \\
        --reference ${reference} \\
        --moving ${moving} \\
        --output ${moving.simpleName}_registered.ome.tiff \\
        --qc-dir qc \\
        --crop-size ${crop_size} \\
        --overlap ${overlap} \\
        --n-features ${n_features} \\
        --n-workers ${n_workers}
    """
}
