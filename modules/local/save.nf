// publish.nf
// Defines a process to transfer results to a slow memory partition

process SAVE {
    tag "Final Publishing"

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

    script:
    """
    echo "Starting rsync transfer of staged results to archive: ${final_archive_dir}"

    # Create destination directory if it doesn't exist
    mkdir -p ${final_archive_dir}/${params.id}_${params.registration_method}/

    # Use rsync for robust transfer
    rsync -avh --progress --timeout=300 results/ ${final_archive_dir}/${params.id}_${params.registration_method}/

    echo "Transfer complete."
    """
}