// publish.nf
// Defines a process to transfer results to a slow memory partition

process SAVE_RESULTS {
    tag "Final Publishing"

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
    
    # Use rsync to copy the contents of the staged directory to the final destination.
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # -v: verbose output
    # -h: human-readable numbers
    # --progress: shows transfer progress (useful for long jobs)
    # --delete: removes files from destination that are not present in source (optional, but good for cleanup)
    # The trailing slash on \${results_dir}/ copies the *contents* of the directory, not the directory itself.
    rsync -avh --progress ${results_dir}/ ${final_archive_dir}/
    
    echo "Transfer complete."
    """
}