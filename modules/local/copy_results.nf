process COPY_RESULTS {
    tag "Copying to savedir"
    label 'process_medium'
    container null

    errorStrategy 'retry'
    maxRetries 3
    time '72h'

    input:
    val(results_ready)
    val(source_dir)
    val(destination_dir)
    val(delete_source)

    output:
    path("transfer.log"), emit: log
    path("verification.log"), emit: verification

    when:
    destination_dir && destination_dir != source_dir

    script:
    def parallel_jobs = task.cpus ?: 4
    """
    #!/bin/bash
    set -euo pipefail

    LOG="transfer.log"
    VERIFY_LOG="verification.log"
    MAX_RETRIES=10
    RETRY_DELAY=120

    exec > >(tee -a \$LOG) 2>&1

    echo "========================================"
    echo "Parallel Copy to Savedir"
    echo "========================================"
    echo "Source:      ${source_dir}"
    echo "Destination: ${destination_dir}"
    echo "Parallel:    ${parallel_jobs} jobs"
    echo "Delete source after copy: ${delete_source}"
    echo "Start:       \$(date)"
    echo ""

    # Pre-flight checks
    if [ ! -d "${source_dir}" ]; then
        echo "ERROR: Source directory does not exist: ${source_dir}"
        exit 1
    fi

    if [ "${source_dir}" = "/" ] || [ -z "${source_dir}" ]; then
        echo "ERROR: Refusing to copy from unsafe source path: '${source_dir}'"
        exit 1
    fi

    if [ "${destination_dir}" = "/" ] || [ -z "${destination_dir}" ]; then
        echo "ERROR: Refusing to copy to unsafe destination path: '${destination_dir}'"
        exit 1
    fi

    if [ "${source_dir}" = "${destination_dir}" ]; then
        echo "ERROR: Source and destination are identical"
        exit 1
    fi

    if [[ "${destination_dir}" == "${source_dir}"/* ]]; then
        echo "ERROR: Destination is nested inside source (would recurse)"
        exit 1
    fi

    SOURCE_SIZE=\$(du -sLb "${source_dir}" | cut -f1)
    SOURCE_FILES=\$(find "${source_dir}" -type f | wc -l)
    echo "Source: \$(numfmt --to=iec \$SOURCE_SIZE) in \$SOURCE_FILES files"

    mkdir -p "${destination_dir}"

    DEST_AVAIL=\$(df -B1 "${destination_dir}" | awk 'NR==2 {print \$4}')
    NEEDED=\$(awk -v size="\$SOURCE_SIZE" 'BEGIN {printf "%.0f", size * 1.05}')  # 5% buffer
    if [ "\$DEST_AVAIL" -lt "\$NEEDED" ]; then
        echo "ERROR: Insufficient space! Available: \$(numfmt --to=iec \$DEST_AVAIL), Need: \$(numfmt --to=iec \$NEEDED)"
        exit 1
    fi
    echo ""

    # Retry loop for transfer
    attempt=1
    SUCCESS=false

    while [ \$attempt -le \$MAX_RETRIES ]; do
        echo "----------------------------------------"
        echo "Transfer attempt \$attempt of \$MAX_RETRIES - \$(date)"
        echo "----------------------------------------"

        # Parallel rsync by top-level items using find (safe for spaces)
        find "${source_dir}" -mindepth 1 -maxdepth 1 -print0 | xargs -0 -I {} -P ${parallel_jobs} \\
            rsync -avL \\
            --inplace \\
            --append-verify \\
            --timeout=600 \\
            --partial \\
            "{}" "${destination_dir}/"

        EXIT_CODE=\$?

        if [ \$EXIT_CODE -eq 0 ]; then
            SUCCESS=true
            break
        fi

        echo "Transfer incomplete (exit \$EXIT_CODE). Retrying in \${RETRY_DELAY}s..."
        sleep \$RETRY_DELAY
        ((attempt++))
    done

    if [ "\$SUCCESS" != true ]; then
        echo "ERROR: Transfer failed after \$MAX_RETRIES attempts"
        exit 1
    fi

    if [ "${delete_source}" = "true" ]; then
        echo ""
        echo "Deleting source: ${source_dir}"
        rm -rf --one-file-system "${source_dir}"
    else
        echo "Source retention is enabled; source directory was not deleted"
    fi

    echo ""
    echo "========================================"
    echo "Complete: \$(date)"
    echo "========================================"
    """

    stub:
    """
    echo "STUB: Would copy data from ${source_dir} to ${destination_dir}" > transfer.log
    echo "STUB: All files verified" > verification.log
    """
}
