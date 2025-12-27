nextflow.enable.dsl = 2

/*
========================================================================================
    PROCESS: VALIDATE_CHECKPOINT
========================================================================================
    Validates checkpoint CSV files to ensure they are correct before resuming pipeline.

    This catches common errors early:
    - Missing required columns
    - Files referenced in CSV don't exist
    - Invalid data formats

    Input:
        checkpoint_csv: Path to the checkpoint CSV file
        step_name: Name of the checkpoint step (preprocessed, registered, postprocessed)

    Output:
        validated: The same CSV file (if validation passes)
========================================================================================
*/

process VALIDATE_CHECKPOINT {
    tag "${step_name}"
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://bolt3x/attend_image_analysis:preprocess' :
        'docker://bolt3x/attend_image_analysis:preprocess' }"

    input:
    path(checkpoint_csv)
    val(step_name)

    output:
    path(checkpoint_csv), emit: validated
    path "validation_report.txt", emit: report

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    #!/usr/bin/env python3
    import pandas as pd
    import sys
    from pathlib import Path

    csv_path = '${checkpoint_csv}'
    step = '${step_name}'

    print(f"========================================")
    print(f"Validating Checkpoint CSV: {step}")
    print(f"========================================\\n")

    # Define required columns for each step
    required_cols = {
        'preprocessed': ['patient_id', 'preprocessed_image', 'is_reference', 'channels'],
        'registered': ['patient_id', 'registered_image', 'is_reference', 'channels'],
        'postprocessed': ['patient_id', 'is_reference', 'phenotype_csv', 'phenotype_mask',
                         'phenotype_mapping', 'merged_csv', 'cell_mask']
    }

    if step not in required_cols:
        print(f"❌ ERROR: Unknown checkpoint step: {step}", file=sys.stderr)
        print(f"Valid steps: {', '.join(required_cols.keys())}", file=sys.stderr)
        sys.exit(1)

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSV file loaded: {len(df)} entries")
    except Exception as e:
        print(f"❌ ERROR: Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate structure
    errors = []
    warnings = []

    for col in required_cols[step]:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        print(f"\\n❌ CSV Structure Errors:")
        for error in errors:
            print(f"  - {error}")
        print(f"\\nFound columns: {', '.join(df.columns)}")
        sys.exit(1)

    print(f"✓ All required columns present")

    # Validate file existence
    print(f"\\nValidating file existence...")
    file_cols = [col for col in df.columns if 'image' in col or 'csv' in col or 'mask' in col or 'mapping' in col]

    missing_files = []
    for idx, row in df.iterrows():
        patient = row['patient_id']
        for col in file_cols:
            if col in row and pd.notna(row[col]):
                file_path = Path(row[col])
                if not file_path.exists():
                    missing_files.append(f"Patient {patient}, {col}: {row[col]}")

    if missing_files:
        print(f"\\n❌ Missing Files ({len(missing_files)}):")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        print(f"\\n💡 Hint: Files may have been deleted or moved since checkpoint creation")
        sys.exit(1)

    print(f"✓ All {len(df) * len(file_cols)} files exist")

    # Validate data types
    print(f"\\nValidating data formats...")
    for idx, row in df.iterrows():
        patient = row['patient_id']

        # Validate is_reference is boolean
        if row['is_reference'] not in ['True', 'False', True, False, 'true', 'false', '1', '0']:
            warnings.append(f"Patient {patient}: is_reference has unexpected value: {row['is_reference']}")

        # Validate channels format (if present)
        if 'channels' in df.columns and pd.notna(row['channels']):
            channels = str(row['channels'])
            if '|' not in channels and ',' not in channels:
                warnings.append(f"Patient {patient}: channels format may be incorrect: {channels}")

    if warnings:
        print(f"\\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings[:5]:
            print(f"  - {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")

    # Summary
    print(f"\\n========================================")
    print(f"✅ Validation PASSED")
    print(f"========================================")
    print(f"Patients: {len(df)}")
    print(f"Reference images: {df['is_reference'].sum() if 'is_reference' in df.columns else 'N/A'}")
    print(f"Step: {step}")
    print(f"CSV: {csv_path}")

    # Write validation report
    with open('validation_report.txt', 'w') as f:
        f.write(f"Checkpoint Validation Report\\n")
        f.write(f"===========================\\n\\n")
        f.write(f"Step: {step}\\n")
        f.write(f"CSV: {csv_path}\\n")
        f.write(f"Patients: {len(df)}\\n")
        f.write(f"Errors: {len(errors)}\\n")
        f.write(f"Warnings: {len(warnings)}\\n")
        f.write(f"Status: PASSED\\n")
    """

    stub:
    """
    touch validation_report.txt
    echo "Checkpoint validation stubbed" > validation_report.txt
    """
}
