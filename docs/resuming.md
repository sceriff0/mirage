# Checkpoints & resuming

<p class="standfirst">A stopped run does not have to start over. Repeat the command with
<code>-resume</code> and finished tasks are reused; or start a later step from the checkpoint CSV an
earlier run wrote.</p>

## The short version

If a run stops, run the same command again with `-resume` added. Finished tasks are not
repeated, as long as the `work/` directory is still there.

## Two mechanisms

!!! note "`-resume` vs `--start` — different mechanisms"
    - `-resume` reuses the **Nextflow work cache**; only changed tasks and their
      descendants re-run. Needs `work/` to still exist.
    - `--start` **skips earlier steps entirely** by feeding a checkpoint CSV. Needs
      only `--outdir`, so it is what you use when `work/` is gone — purged scratch, a
      different machine.

    **Prefer `-resume` when `work/` survives.** Changing a parameter and re-running
    with `-resume` re-runs only the tasks that parameter actually affects, with no
    checkpoint CSV and no `--start`: e.g. changing `--pyramid_resolutions` re-runs the
    pyramid and the exports and reuses the other tasks.

    They compose (`--start postprocessing … -resume`), but understand what you get:
    `--start` reads its inputs from published paths under `--outdir`, which are
    different files from the `work/` originals as far as the cache is concerned, so
    the combination reuses **nothing** from a prior run's work directory. `-resume`
    there only helps across successive `--start postprocessing` runs.

## Checkpoint CSVs { #checkpoints }

Each step ends by writing one **aggregated checkpoint CSV** (covering all patients)
to a single `csv/` folder directly under `--outdir`. Each doubles as the next step's
samplesheet — feed it back in with a matching `--start`. Their headers are a published
contract (`lib/Checkpoint.groovy`).

| File | Written after | Feeds | Columns |
|---|---|---|---|
| `csv/preprocessed.csv` | preprocessing | `--start registration` | `patient_id`, `preprocessed_image`, `is_reference`, `channels` |
| `csv/registered.csv` | registration | `--start segmentation` | `patient_id`, `registered_image`, `is_reference`, `channels` |
| `csv/segmented.csv` | segmentation | `--start postprocessing` | `patient_id`, `registered_image`, `is_reference`, `channels`, `cell_mask`, `nuclei_mask`, `contours`, `nucleus_contours` |
| `csv/postprocessed.csv` | postprocessing | manifest of final outputs | `patient_id`, `cell_csv`, `cell_geojson`, `merged_csv`, `cell_mask`, `pyramid` |

!!! danger "Checkpoints are in `<outdir>/csv/`, NOT `<outdir>/<patient>/csv/`"
    The resume CSVs are aggregated, one file each, directly under `--outdir` — not in
    the per-patient subtree. Keep `--outdir` consistent across steps so resume
    commands point `--input` at the right file.

!!! note "The schema is fixed across parameter settings"
    `nucleus_contours` is empty when `quantify_compartments = false` (the
    extractor does not run), but the **column is still present**. Readers test
    for an empty value, never for a missing column — one header serves every
    run.

## Starting a later step

```bash
# run preprocessing only …
nextflow run . --input samplesheet.csv --outdir results --stop preprocessing -c site.config

# … then pick up from its checkpoint later
nextflow run . --input results/csv/preprocessed.csv --outdir results --start registration -c site.config
```

=== "Resume at registration"

    ```bash
    nextflow run . --input results/csv/preprocessed.csv --outdir results \
      --start registration -profile docker -resume \
      -c site.config
    ```

=== "Resume at segmentation"

    ```bash
    nextflow run . --input results/csv/registered.csv --outdir results \
      --start segmentation -profile docker -resume \
      -c site.config
    ```

=== "Resume at postprocessing"

    ```bash
    nextflow run . --input results/csv/segmented.csv --outdir results \
      --start postprocessing -profile docker -resume \
      -c site.config
    ```

The columns each `--start` requires are in
[The samplesheet → Columns per entry point](samplesheet.md#columns-per-entry-point).

## What can stop a resume from working

!!! warning "`cleanup_work` is ON by default, and it removes the thing `-resume` needs"
    A successful run deletes its work directory, so a later `-resume` has no cache
    to reuse. It does **not** error — the run exits 0 and silently re-runs every
    task. The pipeline **warns at launch** when `cleanup_work` and `-resume` are
    both in play; it does not refuse, because which one you meant is ambiguous.

    While you are iterating with `-resume`, put `{"cleanup_work": false}` in a
    `-params-file` — not in a `-c site.config`, which `nextflow.config`'s `cleanup`
    line cannot see; such a run is refused at launch rather than silently cleaned.
    `--start` restarts are unaffected either way: they read published paths under
    `--outdir`, never `work/` — but see `--cleanup_level` below, because at the
    `final` level those published paths are not written.
    Details: [Output cleanup](parameters.md#output-cleanup).

!!! warning "Checkpoints are written only at `--cleanup_level=none`"
    `none` is the default. A `final` run keeps `pyramid/`, `geojson/`,
    `quantification/`, `spatialdata/` and the QC tree, and publishes nothing
    else — no `registered/`, no `segmentation/`, and no checkpoint CSV at all;
    `<outdir>/csv/` holds a `README.txt` saying so. A manifest whose rows pointed
    at files that were never published would be worse than no manifest: `--start`
    opens exactly what it names.

    So a `final` run's output **cannot be re-entered**. Leave the default (`none`)
    on the run whose output you intend to restart from.
    Details: [Output cleanup](parameters.md#output-cleanup).

!!! warning "Two tasks always re-run"
    `GENERATE_QC_REPORT` and `AGGREGATE_SIZE_LOGS` are declared `cache = false`. Their
    inputs are `collectFile()` results, which are rebuilt into a fresh
    `work/tmp/<hash>/` on every run and so can never match the cache; the QC report
    also embeds a run timestamp. Both are cheap and terminal. Seeing exactly these two
    re-run on an otherwise fully-cached resume is correct — seeing anything *else*
    re-run is a determinism bug (`tests/resume_check.sh` checks precisely that).
