process COPY_RESULTS {
    tag "Copying to savedir"
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:copy' :
        'docker://bolt3x/attend_image_analysis:copy' }"

    input:
    val(results_ready)  // Trigger value to ensure this runs after pipeline completion
    val(source_dir)
    val(destination_dir)

    output:
    path("rsync.log"), emit: log

    when:
    destination_dir && destination_dir != source_dir

    script:
    """
    echo "========================================" > rsync.log
    echo "Copying pipeline results to savedir" >> rsync.log
    echo "========================================" >> rsync.log
    echo "Source:      ${source_dir}" >> rsync.log
    echo "Destination: ${destination_dir}" >> rsync.log
    echo "Start time:  \$(date)" >> rsync.log
    echo "" >> rsync.log

    # Create destination directory if it doesn't exist
    mkdir -p ${destination_dir}

    # Rsync with:
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # -v: verbose
    # -L: dereference symlinks (copy actual files, not links)
    # --update: skip files newer in destination
    # --info=progress2: show overall progress
    rsync \\
        -avL \\
        --update \\
        --info=progress2 \\
        --log-file=rsync_detailed.log \\
        ${source_dir}/ \\
        ${destination_dir}/ 2>&1 | tee -a rsync.log

    EXIT_CODE=\${PIPESTATUS[0]}

    echo "" >> rsync.log
    echo "End time:    \$(date)" >> rsync.log
    echo "Exit code:   \$EXIT_CODE" >> rsync.log

    if [ \$EXIT_CODE -eq 0 ]; then
        echo "Status:      SUCCESS" >> rsync.log
        echo "" >> rsync.log
        echo "Results successfully copied to ${destination_dir}" >> rsync.log
    else
        echo "Status:      FAILED" >> rsync.log
        echo "" >> rsync.log
        echo "ERROR: Rsync failed with exit code \$EXIT_CODE" >> rsync.log
        echo "Check rsync_detailed.log for details" >> rsync.log
        exit \$EXIT_CODE
    fi

    # Append detailed log for debugging
    echo "" >> rsync.log
    echo "========================================" >> rsync.log
    echo "Detailed rsync log:" >> rsync.log
    echo "========================================" >> rsync.log
    cat rsync_detailed.log >> rsync.log
    """

    stub:
    """
    echo "STUB MODE: Would copy results" > rsync.log
    echo "Source:      ${source_dir}" >> rsync.log
    echo "Destination: ${destination_dir}" >> rsync.log
    """
}
