/*
 * SAVE - Copy results to archive storage
 *
 * Transfers pipeline results to a final archive location using rsync.
 * Designed for I/O-heavy transfers to slow storage partitions (e.g., NFS).
 *
 * Input: Results files to archive and destination path
 * Output: versions.yml only (files are copied to archive)
 */
process SAVE {
    tag "Final Publishing"
    label 'process_single'

    // Disable container for this process - run natively to access NFS mounts
    container null

    // IMPORTANT: Set generous resources for this I/O-heavy step
    // Adjust time and memory based on the expected size of your 'results' directory
    time '6h'
    cpus 4
    memory '64 GB'

    // Use a queue or partition optimized for data transfer, if available
    // queue 'transfer_partition'

    // Define the required input
    input:
    path results_files, stageAs: 'results/*' // Individual files to save, staged into results/ dir
    val final_archive_dir // The final destination path (e.g., /hpcnfs/results/)

    output:
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "Starting rsync transfer of staged results to archive: ${final_archive_dir}"

    # Create destination directory if it doesn't exist
    mkdir -p ${final_archive_dir}/${params.id}_${params.registration_method}/

    # Use rsync for robust transfer (-L follows symlinks)
    rsync -rL --progress results/ ${final_archive_dir}/${params.id}_${params.registration_method}/

    echo "Transfer complete."

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        rsync: \$(rsync --version | head -n1 | sed 's/rsync  version //' | cut -d' ' -f1)
    END_VERSIONS
    """

    stub:
    """
    echo "STUB: Would transfer to ${final_archive_dir}/${params.id}_${params.registration_method}/"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        rsync: stub
    END_VERSIONS
    """
}