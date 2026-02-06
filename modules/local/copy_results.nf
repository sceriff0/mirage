process COPY_RESULTS {
    tag "Copying to savedir"
    label 'process_medium'  // Need more CPUs for parallel
    container null

    errorStrategy 'retry'
    maxRetries 3
    time '72h'

    input:
    val(results_ready)
    val(source_dir)
    val(destination_dir)

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
    echo "Start:       \$(date)"
    echo ""

    # Pre-flight checks
    SOURCE_SIZE=\$(du -sb ${source_dir} | cut -f1)
    SOURCE_FILES=\$(find ${source_dir} -type f | wc -l)
    echo "Source: \$(numfmt --to=iec \$SOURCE_SIZE) in \$SOURCE_FILES files"

    mkdir -p ${destination_dir}

    DEST_AVAIL=\$(df -B1 ${destination_dir} | awk 'NR==2 {print \$4}')
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

        # Parallel rsync by top-level items using xargs
        ls ${source_dir} | xargs -I {} -P ${parallel_jobs} \\
            rsync -avL \\
            --inplace \\
            --append-verify \\
            --timeout=600 \\
            --partial \\
            ${source_dir}/{} ${destination_dir}/

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

    echo ""
    echo "========================================"
    echo "Verifying transfer with checksums"
    echo "========================================"

    cd ${source_dir}

    # Use BLAKE3 if available (5-10x faster than MD5), fallback to MD5
    if command -v b3sum &> /dev/null; then
        echo "Using BLAKE3 (parallel, fast)..."
        find . -type f -print0 | xargs -0 -P ${parallel_jobs} -I {} b3sum {} > /tmp/checksums.txt
        cd ${destination_dir}
        b3sum -c /tmp/checksums.txt > \$VERIFY_LOG 2>&1
    else
        echo "Using MD5 (parallel)..."
        find . -type f -print0 | xargs -0 -P ${parallel_jobs} -I {} md5sum {} > /tmp/checksums.txt
        cd ${destination_dir}
        md5sum -c /tmp/checksums.txt > \$VERIFY_LOG 2>&1
    fi

    VERIFY_EXIT=\$?

    if [ \$VERIFY_EXIT -ne 0 ] || grep -q "FAILED" \$VERIFY_LOG; then
        echo ""
        echo "ERROR: Checksum verification failed!"
        grep "FAILED" \$VERIFY_LOG || true
        echo "Source NOT deleted"
        exit 1
    fi

    echo ""
    echo "All \$SOURCE_FILES files verified successfully"
    echo ""
    echo "Deleting source: ${source_dir}"
    rm -rf ${source_dir}

    echo ""
    echo "========================================"
    echo "Complete: \$(date)"
    echo "========================================"
    """

    stub:
    """
    echo "STUB: Would parallel copy data" > transfer.log
    echo "STUB: All files OK" > verification.log
    """
}
