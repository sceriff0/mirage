# Your first run

<p class="standfirst">From a computer with almost nothing installed to MIRAGE running on your own
slides at its defaults. Five steps, about ten minutes of your time; after that the computer works on
its own.</p>

You do not need to know Nextflow, Python or Docker. You will copy four commands into a terminal and
fill in one small table.

!!! info "What ten minutes covers"
    Ten minutes gets the run **started**. The first run then downloads the tools (several
    gigabytes, once: about 18 GB at the defaults), and the analysis itself takes from minutes for a small image to hours for
    a full cohort of whole slides.

## What you need

| | |
|---|---|
| **A computer** | With an Intel or AMD processor: Linux, Windows with [WSL 2](https://learn.microsoft.com/windows/wsl/install) (open the "Ubuntu" terminal and follow the Linux instructions), or an Intel Mac. A cluster works too — see step 4. On an Apple-silicon Mac (M1 and later) a run does not finish — see the note below. [Install](installation.md) has the per-platform details. |
| **Your slides** | One image file per staining round, each containing the same nuclear stain (DAPI). See [which formats are read](validation/format_validation.md). |
| **The channel names** | For each file, the names of its channels in the order they were acquired. |
| **Disk space** | About 40 GB for the tools with Docker (about 20 GB with Apptainer or Singularity), plus room for results: several times the size of your slides. |

!!! warning "How much memory"
    Registration with STARE needs under 4 GB per task whatever the slide size. Segmentation and
    quantification do not have that bound: they need memory in proportion to the slide. A laptop
    with 16 GB handles small sections and test crops; whole-slide cohorts belong on a workstation
    with 64 GB or more, or on a cluster. [Resources](resources.md) has the numbers.

!!! warning "Apple-silicon Macs (M1 and later)"
    The tools are built for Intel and AMD processors. On an Apple-silicon Mac, Docker runs them
    through emulation: on an M2 Mac with Docker Desktop's default (QEMU) emulation, conversion
    and STARE registration ran but segmentation with StarDist crashed (exit code 139), so a run
    does not finish. Docker Desktop's *Use Rosetta for x86_64/amd64 emulation* setting has not
    been tested with MIRAGE. The reliable route is an Intel or AMD machine, or a cluster — see
    [Install → macOS](installation.md#macos).

## 1. Install three things

Open a terminal (on macOS: the *Terminal* app) and paste each block.

**Java** — Nextflow runs on it.

```bash
curl -s https://get.sdkman.io | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java
```

**Nextflow** — the program that runs the pipeline.

```bash
curl -s https://get.nextflow.io | bash
chmod +x nextflow
mkdir -p "$HOME/.local/bin" && mv nextflow "$HOME/.local/bin/"
export PATH="$HOME/.local/bin:$PATH"
```

**Docker** — every analysis tool (STARE, StarDist, Bio-Formats, …) comes pre-packaged in a
container, so you install none of them yourself. Install
[Docker Desktop](https://docs.docker.com/get-docker/) and start it. In *Settings → Resources*,
give it as much memory as your computer can spare.

Check that all three answer:

```bash
java -version
nextflow -version
docker run --rm hello-world
```

`nextflow -version` must report **25.04.0 or newer**.

??? note "On a cluster"
    Clusters usually provide Java as a module (`module load java`) and Apptainer or Singularity
    instead of Docker. Install only Nextflow, as above, and use the cluster command in step 4.

## 2. Get MIRAGE

```bash
git clone https://github.com/sceriff0/mirage.git
cd mirage
```

Run every command below from inside this `mirage` folder.

## 3. Describe your slides

MIRAGE reads one small table, the **samplesheet**, with one row per image file. Make it in a
spreadsheet program and save it as `samplesheet.csv` (comma-separated) inside the `mirage` folder.

| patient_id | path_to_file | is_reference | channels |
|---|---|---|---|
| P001 | /data/P001_round1.nd2 | true | DAPI\|PANCK\|SMA |
| P001 | /data/P001_round2.nd2 | false | DAPI\|CD45\|CD163 |
| P001 | /data/P001_round3.nd2 | false | DAPI\|CD8\|CD4 |

As a file, that is:

```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_round1.nd2,true,DAPI|PANCK|SMA
P001,/data/P001_round2.nd2,false,DAPI|CD45|CD163
P001,/data/P001_round3.nd2,false,DAPI|CD8|CD4
```

Four rules (the full contract is in [The samplesheet](samplesheet.md)):

1. **`patient_id`** — the same label on every round of one section. Each patient is processed on
   its own.
2. **`path_to_file`** — the full path to the image, starting with `/`.
3. **`is_reference`** — `true` on exactly one row per patient. Every other round is aligned onto
   this one, so pick a round with a clean, complete nuclear stain.
4. **`channels`** — the channel names in acquisition order, separated by `|`. Every row must
   contain the nuclear stain (`DAPI`). Apart from the nuclear stain, a marker name may appear only
   once per patient.

## 4. Run

=== "Laptop or workstation"

    ```bash
    nextflow run . -profile local,docker \
      -params-file params/no_gpu.json \
      --input samplesheet.csv --outdir results
    ```

    `local` caps the run at 4 CPUs and 16 GB. `params/no_gpu.json` runs segmentation on the
    processor; leave that line out on a Linux machine with an NVIDIA GPU
    ([GPU or no GPU](gpu.md)). For a bigger
    workstation, [size the run to your machine](hpc.md#size-your-run).

=== "Cluster (SLURM)"

    Tell MIRAGE how large one job may be, once:

    ```bash
    cp conf/site.config.template site.config
    ```

    Open `site.config` in a text editor and set `max_cpus`, `max_memory` and, if your cluster
    needs them, `slurm_partition` and `slurm_account`. Then:

    ```bash
    nextflow run . -profile slurm,singularity -c site.config \
      --input samplesheet.csv --outdir results
    ```

    Start it inside `tmux` or `screen`, or as a job of its own, so it survives your logging out.
    More in [Running on a cluster](hpc.md).

A list of steps appears and each is ticked (✔) as it finishes. The run is complete when the
terminal prompt returns with every step ticked and no error message. If it stops with an error,
run the same command again with `-resume` added: finished steps are not repeated.

## 5. Look at the results

Everything is in `results/`, one folder per patient:

| Open this | In | To see |
|---|---|---|
| `results/qc/` → the HTML report | a web browser | whether registration and segmentation worked |
| `results/P001/pyramid/pyramid.ome.tiff` | [QuPath](https://qupath.github.io/) | all rounds aligned into one multi-channel image |
| `results/P001/geojson/export/cells.geojson` | QuPath, dragged onto the open image | every cell, with its marker intensities |
| `results/P001/quantification/merged_quant.csv` | a spreadsheet, R or Python | one row per cell |
| `results/P001/spatialdata/P001.zarr` | [scverse](https://scverse.org/) tools | the same cells as a SpatialData store |

To turn intensities into cell types, gate the cells in QuPath with
[FlowPath](https://flowpath.readthedocs.io/).

**Look at the registration first.** Open the QC report and check the before/after overlays of the
nuclear stain: after registration the two colours should coincide.
[Checking registration](registration_qc.md) explains each number.

## What ran

You changed nothing, so the run used the defaults — the configuration of the paper:

| Step | Default | To change it |
|---|---|---|
| Conversion | every format read to OME-TIFF, nuclear stain moved to the first channel | — |
| Illumination correction | **off** | [Parameters → Preprocessing](parameters.md#preprocessing) |
| Registration | **STARE**: tile-based non-rigid alignment on the nuclear stain | `--registration_method valis` |
| Segmentation | **StarDist** with MIRAGE's trained nuclear model; each cell is its nucleus grown by 10 px | `--seg_method instantseg` or `cellsam` |
| Quantification | median intensity of every marker in nucleus, cytoplasm and whole cell | [Parameters → Quantification](parameters.md#quantification) |
| Export | pyramidal OME-TIFF, QuPath GeoJSON, SpatialData store, QC report | [Parameters → Visualization & export](parameters.md#visualization-export) |

## If it stops

Run the same command again with `-resume` added: finished steps are not repeated
([Checkpoints & resuming](resuming.md)). The common first-run errors — missing
`max_cpus`/`max_memory`, a missing pixel size, a refused samplesheet, Docker not running,
a step out of memory — and their fixes are in
[Troubleshooting → First run](troubleshooting.md#first-run).

## No slides at hand?

MIRAGE can make a small synthetic dataset and run on it:

```bash
python tests/testdata/generate_complete_testdata.py
nextflow run . -profile test,docker --outdir results
```

The first line needs Python with three packages (`pip install numpy tifffile h5py`).

The `test` profile is a small fixture, not the shipped defaults: it pins VALIS and a built-in
StarDist model and runs segmentation on the CPU, so that it finishes in a few minutes on
synthetic images.
