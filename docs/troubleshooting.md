# Troubleshooting

## `Invalid --step` or `Invalid --registration_method`

Use only supported enums:

- Step: `preprocessing`, `registration`, `postprocessing`, `copy_results`
- Registration: `valis`, `valis_pairs`, `gpu`, `cpu`, `cpu_tiled`

## `--input` Validation Errors

`--input` is required for:

- `preprocessing`
- `registration`
- `postprocessing`

`copy_results` does not use CSV input.

## `copy_results` Path Errors

Common causes:

- Missing `--savedir`
- `savedir` equals `outdir`
- `savedir` nested inside `outdir`
- Missing source directory

## Out-of-Memory / Runtime Failures

Actions:

1. Lower memory pressure by switching method (for example `valis` -> `gpu` or `cpu_tiled`).
2. Increase global caps (`max_memory`, `max_time`).
3. Check process-specific settings in `conf/modules.config`.
4. Resume with `-resume` after adjustments.

## Nextflow Not Found

If `nextflow` is not installed or not in `PATH`, install/activate it and retry:

```bash
nextflow -version
```

## Debug Channel Visibility

Set:

```bash
--debug_channels true
```

to enable channel-level debug `.view` output in subworkflows where supported.

