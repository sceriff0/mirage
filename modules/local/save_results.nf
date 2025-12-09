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

    # Use rsync if available, otherwise fall back to cp
    # rsync is more robust for large files and network filesystems
    if command -v rsync &> /dev/null; then
        echo "Using rsync for robust transfer..."
        rsync -avh --progress --timeout=300 ${results_dir}/ ${final_archive_dir}/
    else
        echo "rsync not found, using cp..."
        # Copy files with error handling
        # Process large files separately to isolate I/O errors
        find ${results_dir} -type f -size +1G -print0 | while IFS= read -r -d '' large_file; do
            rel_path=\${large_file#${results_dir}/}
            dest_file="${final_archive_dir}/\${rel_path}"
            mkdir -p "\$(dirname "\${dest_file}")"
            echo "Copying large file: \${rel_path}"

            # Use dd for large files with network-friendly block size
            if ! dd if="\${large_file}" of="\${dest_file}" bs=1M status=progress; then
                echo "WARNING: Failed to copy \${rel_path}" >&2
                # Continue with other files
            fi
        done

        # Copy remaining smaller files
        find ${results_dir} -type f -size -1G -print0 | while IFS= read -r -d '' file; do
            rel_path=\${file#${results_dir}/}
            dest_file="${final_archive_dir}/\${rel_path}"
            mkdir -p "\$(dirname "\${dest_file}")"
            if ! cp -p "\${file}" "\${dest_file}"; then
                echo "WARNING: Failed to copy \${rel_path}" >&2
            fi
        done
    fi

    echo "Transfer complete."
    """
}