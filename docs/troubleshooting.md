# Troubleshooting

<p class="standfirst">The errors people actually hit, what each one means, and the fix. After any
fix, run the same command again with <code>-resume</code>: finished steps are not repeated.</p>

## First run

??? failure "`Missing required parameter(s): max_cpus, max_memory`"
    The command has neither `-profile local` nor `-c site.config`. Add one of them —
    see [Your first run](quickstart.md#4-run) and
    [Make a site config](hpc.md#size-your-run).

??? failure "The run stops at the start with a message about pixel size"
    Your files do not record their pixel size. Add it to the command, in micrometres per pixel,
    for example `--pixel_size 0.325`.

??? failure "The run refuses the samplesheet"
    The message names the row. The usual causes are no `is_reference=true` row for a patient,
    two of them, a row without the nuclear stain, or the same marker name on two rounds. See
    [The samplesheet → Rules](samplesheet.md#rules).

??? failure "Docker: `permission denied` or `Cannot connect to the Docker daemon`"
    Docker Desktop is not running, or on Linux your user is not in the `docker` group.

??? failure "Docker: `could not select device driver \"\" with capabilities: [[gpu]]`"
    Segmentation asks for a GPU by default and this machine has none Docker can use (any
    Mac, most laptops, WSL without the NVIDIA toolkit). Add
    `-params-file params/no_gpu.json` to run segmentation on the CPU. See [GPU](gpu.md).

??? failure "Segmentation crashes with exit code 139 on an Apple-silicon Mac"
    The images are built for Intel/AMD processors and run under emulation on Apple
    silicon; StarDist segmentation crashes there. See
    [Install → macOS](installation.md#macos) for the options.

??? failure "The run seems to hang at startup"
    Usually Nextflow pulling large container images on first run (about 18 GB at the
    defaults). Pre-pull them once with the list in
    [Running on a cluster → Pre-pulling container images](hpc.md#pre-pulling-container-images).

## Launch validation

??? failure "Launch fails on an invalid value"
    `--start`/`--stop` must be `preprocessing|registration|segmentation|postprocessing`;
    `--registration_method` must be `valis|tiled`; `--seg_method` must be
    `stardist|instantseg|cellsam`. Typos exit before any process is submitted.

??? failure "`--input` validation error / wrong columns"
    The most common mistake is feeding the **preprocessing** samplesheet to
    `--start registration`. Each step needs its own column (`path_to_file` →
    `preprocessed_image` → `registered_image`; `registration` and `segmentation`
    both key on `registered_image`, and `--start postprocessing` additionally
    requires `cell_mask` and `nuclei_mask`, because `segmented.csv` carries the
    segmentation masks alongside the image path). Feed the checkpoint CSV that
    matches the step you're resuming — see
    [Columns per entry point](samplesheet.md#columns-per-entry-point).

??? failure "`expanded_quantification requires quantify_compartments to also be true`"
    Expanded output depends on compartments. Both are booleans, so set them in a
    `-params-file`: either add `"quantify_compartments": true` alongside
    `"expanded_quantification": true`, or drop `expanded_quantification` for a
    flat per-cell table. Neither can be passed on the command line — see
    [Boolean parameters](usage.md#boolean-parameters).

## Memory and time

??? failure "Out-of-memory / exit code 137 or 140"
    The job exceeded its memory or time grant. On a laptop, close other programs and
    raise Docker Desktop's memory; beyond that the slide needs a larger machine. On a
    cluster, raise `max_memory`, `max_cpus`, `max_time` in `site.config`. MIRAGE
    auto-retries on `104,134,135,137,139,140,143` with scaled resources. For VALIS
    registration, which runs at valis-wsi's defaults and has no size settings,
    `REGISTER` doubles its memory on each retry up to 512 GB; on a smaller node use
    `--registration_method tiled` (STARE, the default), whose every step is bounded by
    a parameter. See [Resources](resources.md).

## Singularity / Apptainer

??? failure "Singularity: `conveyor failed to get: unexpected end of JSON input`"
    Two Apptainer processes wrote the same download into its shared cache at the same
    time, and one read a file the other was still writing. The task dies before its
    script runs. Apptainer's cache is not built for parallel use.

    The usual cause is `ociAutoPull = true`, which makes every task convert its own
    image: dozens of tasks starting together on a cold cache collide. The `singularity`
    profile leaves it off, like nf-core's, so the Nextflow head pulls each image once
    into `singularity.cacheDir` (or `NXF_SINGULARITY_CACHEDIR`) and tasks only read the
    finished file. Check that no site or `-c` config turns it on.

    Head pulls can still overlap, less often: one head pulls *different* images at the
    same time, and some images share base layers (`mirage-quantify` and `mirage-regqc`
    do); several Nextflow runs started at once against the same `cacheDir` can also
    pull the same image together. To rule that out, launch Nextflow with Apptainer's
    cache disabled. Each pull then unpacks into its own temporary folder, and Nextflow
    still keeps the finished image in `cacheDir`, so later runs download nothing:
    ```bash
    export APPTAINER_DISABLE_CACHE=true SINGULARITY_DISABLE_CACHE=true
    # unpacked layers go here during a pull (tens of GB for the CUDA images)
    export APPTAINER_TMPDIR=/shared/scratch/apptainer_tmp SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
    ```
    After a failure, clear the damaged entries with `singularity cache clean`, then
    `-resume`.

??? failure "Singularity pull: `mksquashfs command failed: signal: killed`"
    The Nextflow head converts every image that is not yet in `singularity.cacheDir`,
    and it converts different images at the same time. `mksquashfs` sizes its memory
    from the node's *physical* RAM, not from the job's limit, so on a cold cache a
    head job with a small memory grant can be killed. This costs memory on the first
    run only: once the images are in the cache, no later run converts anything. Give
    the head job more memory (or run it where it is not memory-capped) for that first
    run. A cluster administrator can also cap the tool with `mksquashfs mem` in
    `apptainer.conf`. Raising `singularity.pullTimeout` does not help. Do not set
    `ociAutoPull = true` to avoid this; it trades this failure for the one above.

??? failure "Singularity: `FATAL: ... permission denied`"
    The cache isn't writable. Point it at a path you own:
    ```bash
    export NXF_SINGULARITY_CACHEDIR=$HOME/.singularity_cache
    export SINGULARITY_CACHEDIR=$HOME/.singularity_cache
    ```

??? failure "`Plugin with id nf-schema not found in any repository`"
    The node has no network access. See
    [Offline / air-gapped execution](hpc.md#offline-air-gapped-execution).

## Resuming

??? tip "Resetting a stuck task"
    If `-resume` keeps re-running the same failing task after you've fixed params,
    clear just that task's cache by hash (shown in the error): `rm -rf work/<hash>*`
    then resume. Avoid `rm -rf work/` wholesale.

??? question "`-resume` re-ran everything"
    A successful run deletes `work/` by default (`cleanup_work`), so there was nothing
    to reuse. See [Checkpoints & resuming](resuming.md#what-can-stop-a-resume-from-working).

## Still stuck?

Open an issue with your command line, the relevant `.nextflow.log` excerpt, and the
failing task's `.command.log` on the
[GitHub tracker](https://github.com/sceriff0/mirage/issues).
