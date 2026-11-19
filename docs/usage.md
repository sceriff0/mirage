# Command line

<p class="standfirst">The flags a run takes, how to pass true/false settings, the command shapes
for common jobs, and what launch validation checks before anything runs.</p>

New here? [Your first run](quickstart.md) is the short path. For every setting see
[Parameters](parameters.md).

!!! note "Two kinds of flags"
    Nextflow distinguishes **pipeline parameters** (double dash, `--input`) from
    **Nextflow options** (single dash, `-profile`, `-resume`, `-c`,
    `-params-file`). Both appear on the same command line.

## The minimal command

```bash
nextflow run . --input samplesheet.csv --outdir results -profile docker -c site.config
```

`--input`, `--outdir`, `--max_cpus` and `--max_memory` are all **required**; the
last two come from `site.config` (see [Make a site config](hpc.md#size-your-run)) or
from `-profile local`. `-profile` selects execution + container profiles,
comma-combined — see [Execution profiles](hpc.md#execution-profiles).

MIRAGE runs four steps, always in this order: `preprocessing`, `registration`,
`segmentation`, `postprocessing` ([The steps](pipeline.md)). `--start` and `--stop`
choose where to enter and exit; omit `--stop` to run to the end, and use
`--start X --stop X` for exactly one step.

## Runtime flags

| Flag | Kind | Required | Description |
|---|---|:---:|---|
| `--input` | param | yes | Samplesheet CSV. Columns depend on `--start` — see [The samplesheet](samplesheet.md#columns-per-entry-point). |
| `--outdir` | param | yes | Output root directory. Checkpoints are written to `<outdir>/csv/`. |
| `--start` | param | no | Entry stage: `preprocessing` (default), `registration`, `segmentation`, `postprocessing`. |
| `--stop` | param | no | Last stage to run. Omitted = run to the end. |
| `dry_run` | param | no | Validate inputs and the samplesheet, then exit without running tasks. Boolean — set it in a `-params-file` (`params/dry_run.json`), never on the CLI; see [Boolean parameters](#boolean-parameters). |
| `--cleanup_level` | param | no | Which published outputs to keep: `none` (default — everything, re-enterable by `--start`) or `final` (final artifacts only; intermediates are never published). |
| `cleanup_work` | param | no | Delete `work/` after a successful run. **On by default**; incompatible with `-resume`. Boolean — set it in a `-params-file`, never on the CLI; see [Boolean parameters](#boolean-parameters). |
| `-profile` | option | no | Execution/config profiles, comma-combined (e.g. `slurm,singularity`). |
| `-params-file` | option | no | JSON preset of parameters, e.g. `params/full_pipeline.json`. |
| `-resume` | option | no | Reuse cached results from a previous run's `work/`. |
| `-c` | option | yes on a cluster | Layer your `site.config` (required `max_cpus`/`max_memory`, SLURM fields). See [Make a site config](hpc.md#size-your-run). |

For the complete parameter list see [Parameters](parameters.md).

## Boolean parameters

**A boolean parameter cannot be passed on the command line.** Set it in a
`-params-file` or a profile instead:

```bash
# works
nextflow run . --input samplesheet.csv --outdir results -c site.config -params-file params/dry_run.json

nextflow run . --input samplesheet.csv --dry_run true   # x fails on Nextflow 26
nextflow run . --input samplesheet.csv --dry_run        # x fails on Nextflow 26
```

Nextflow 26 hands every `--param` to the schema validator as a **string**, so a
boolean arrives as `"true"` rather than `true` and validation rejects it:

```
* --dry_run (true): Value is [string] but should be [boolean]
```

Both CLI forms fail — `--flag true` and a bare valueless `--flag` alike (on
Nextflow 25 a bare flag still arrives as a boolean; on 26 it does not). This
applies to every boolean parameter in the schema, not just `dry_run`, and it is
why the examples in these docs write booleans as `name = value` (params-file /
config syntax) rather than as flags.

A `-params-file` carries a real JSON boolean, which takes a different code path
and works on both engines.

## Common invocations

=== "Full pipeline"

    ```bash
    nextflow run . --input samplesheet.csv --outdir results \
      --start preprocessing -profile docker \
      -params-file params/full_pipeline.json \
      -c site.config
    ```

=== "Single stage"

    ```bash
    nextflow run . --input samplesheet.csv --outdir results \
      --start preprocessing --stop preprocessing -profile docker \
      -c site.config
    ```

=== "Dry run"

    ```bash
    nextflow run . --input samplesheet.csv --outdir results \
      --start preprocessing -params-file params/dry_run.json \
      -c site.config
    ```

To restart from a checkpoint, see [Checkpoints & resuming](resuming.md#starting-a-later-step).

JSON presets in `params/` (`full_pipeline.json`, `preprocessing_only.json`,
`registration_only.json`, `postprocessing_only.json`, `no_gpu.json`, `dry_run.json`,
`test.json`) set ready-made values; load one with `-params-file` and override values
inline — see [Parameter presets](parameters.md#parameter-presets).

## Parameter validation

Every parameter's type, allowed values and range is declared once in
`nextflow_schema.json` and enforced at launch by the
[nf-schema](https://nextflow-io.github.io/nf-schema/) plugin, pinned to
`nf-schema@2.5.1` in `nextflow.config`. A bad value stops the run before any
process is submitted, and every violation is reported at once, by flag name:

```text
ERROR ~ Validation of pipeline parameters failed!

The following invalid input values have been detected:

* --reg_qc (7): Expected any of [[0, 1, 2]]
* --seg_method (notamethod): Expected any of [[stardist, instantseg, cellsam]]
```

Validation is strict about **types**, which matters most with `-params-file`.
JSON lets you write `"skip_preprocess_qc": "false"` — a *string*, not a boolean,
and every non-empty string is truthy in Groovy, so it would turn the switch **on**.
Validation rejects it:

```text
* --skip_preprocess_qc (false): Value is [string] but should be [boolean]
```

Write booleans unquoted in a params file: `"skip_preprocess_qc": false`. The
same applies to numbers — a quoted `"reg_qc": "2"` is a string too and is
rejected rather than coerced.

Rules a JSON Schema cannot express — `--stop` must not precede `--start`,
`--expanded_quantification` requires `--quantify_compartments`, samplesheet
semantics — are checked in `lib/` and fire
right after schema validation.
