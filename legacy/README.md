# Legacy Artifacts

This directory contains modules and tests that are no longer part of the active
pipeline execution surface.

Archived in this refactor:

- `legacy/modules/local/conversion.nf`
- `legacy/modules/local/compute_features.nf`
- `legacy/modules/local/get_preprocess_dir.nf`
- `legacy/modules/local/save.nf`
- `legacy/modules/local/merge.nf`
- `legacy/modules/local/write_checkpoint_csv.nf`
- `legacy/tests/modules/compute_features.nf.test`
- `legacy/tests/modules/merge.nf.test`
- `legacy/tests/modules/write_checkpoint_csv.nf.test`

They are retained for historical reference only and are not imported by
`main.nf` or active subworkflows.
