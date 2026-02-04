process COPY_RESULTS {
    tag "Copying to savedir"
    label 'process_single'
    container null
    
    errorStrategy 'terminate'
    time '72h'  // 10TB of huge files needs time
    
    input:
    val(results_ready)
    val(source_dir)
    val(destination_dir)
    
    output:
    path("rsync.log"), emit: log
    
    when:
    destination_dir && destination_dir != source_dir
    
    script:
    """
    #!/bin/bash
    set -uo pipefail
    
    LOG="rsync.log"
    MAX_RETRIES=50
    RETRY_DELAY=180
    
    exec > >(tee -a \$LOG) 2>&1
    
    echo "========================================"
    echo "Copying pipeline results to savedir"
    echo "========================================"
    echo "Source:      ${source_dir}"
    echo "Destination: ${destination_dir}"
    echo "Start time:  \$(date)"
    echo ""
    
    # List what we're transferring
    echo "Files to transfer:"
    ls -lhS ${source_dir}/ | head -20
    echo ""
    
    SOURCE_SIZE=\$(du -sh ${source_dir} | cut -f1)
    SOURCE_FILES=\$(find ${source_dir} -type f | wc -l)
    echo "Total: \$SOURCE_SIZE in \$SOURCE_FILES files"
    echo ""
    
    mkdir -p ${destination_dir}
    
    # Check destination space
    DEST_AVAIL=\$(df -BG ${destination_dir} | awk 'NR==2 {print \$4}' | tr -d 'G')
    SOURCE_GB=\$(du -BG -s ${source_dir} | cut -f1 | tr -d 'G')
    echo "Destination available: \${DEST_AVAIL}G, Need: \${SOURCE_GB}G"
    
    if [ "\$DEST_AVAIL" -lt "\$SOURCE_GB" ]; then
        echo "ERROR: Insufficient space at destination!"
        exit 1
    fi
    echo ""
    
    # Retry loop
    attempt=1
    SUCCESS=false
    
    while [ \$attempt -le \$MAX_RETRIES ]; do
        echo "----------------------------------------"
        echo "Attempt \$attempt of \$MAX_RETRIES - \$(date)"
        echo "----------------------------------------"
        
        # Key flags for huge files over NFS:
        # --inplace: write directly to dest file (avoids temp file close/rename issues)
        # --append-verify: resume huge files from where they left off + verify
        # --timeout: don't hang on stalled NFS
        # --bwlimit: don't overwhelm NFS (adjust based on your network)
        
        rsync \\
            -avL \\
            --inplace \\
            --append-verify \\
            --timeout=600 \\
            --bwlimit=200000 \\
            --progress \\
            --stats \\
            ${source_dir}/ \\
            ${destination_dir}/
        
        EXIT_CODE=\$?
        
        echo ""
        echo "Rsync exit code: \$EXIT_CODE"
        
        if [ \$EXIT_CODE -eq 0 ]; then
            SUCCESS=true
            break
        fi
        
        echo "Failed. Waiting \${RETRY_DELAY}s before retry..."
        sleep \$RETRY_DELAY
        ((attempt++))
    done
    
    echo ""
    echo "========================================"
    echo "End time: \$(date)"
    
    if [ "\$SUCCESS" = true ]; then
        echo "Status: SUCCESS"
        echo ""
        
        # Verify with checksums for huge files (most reliable)
        echo "Verifying transfer with checksums..."
        VERIFY_FAILED=false
        
        cd ${source_dir}
        for f in *; do
            if [ -f "\$f" ]; then
                echo -n "  Checking \$f... "
                SRC_SUM=\$(md5sum "\$f" | cut -d' ' -f1)
                DST_SUM=\$(md5sum "${destination_dir}/\$f" | cut -d' ' -f1)
                if [ "\$SRC_SUM" = "\$DST_SUM" ]; then
                    echo "OK"
                else
                    echo "MISMATCH!"
                    VERIFY_FAILED=true
                fi
            fi
        done
        
        echo ""
        if [ "\$VERIFY_FAILED" = true ]; then
            echo "ERROR: Checksum verification failed!"
            echo "Source NOT deleted"
            exit 1
        fi
        
        echo "All checksums verified"
        echo ""
        echo "Deleting source: ${source_dir}"
        rm -rf ${source_dir}
        echo "Source deleted successfully"
    else
        echo "Status: FAILED after \$MAX_RETRIES attempts"
        exit 1
    fi
    """
    
    stub:
    """
    echo "STUB MODE: Would copy ~10TB" > rsync.log
    """
}