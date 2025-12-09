// publish.nf
// Defines a process to transfer results to a slow memory partition

process SAVE_RESULTS {
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
    path results_dir // The 'results' directory containing all staged files
    val final_archive_dir // The final destination path (e.g., /hpcnfs/results/)

    script:
    """
    echo "Starting slow transfer of staged results to archive: ${final_archive_dir}"

    # Create destination directory if it doesn't exist
    mkdir -p ${final_archive_dir}

    # Use cp to copy the contents of the staged directory to the final destination.
    # -r: recursive (copy directories)
    # -p: preserve permissions, timestamps, etc.
    # -v: verbose output
    # The trailing slash approach doesn't work with cp, so we use wildcard
    cp -rpv ${results_dir}/* ${final_archive_dir}/

    echo "Transfer complete."
    """
}