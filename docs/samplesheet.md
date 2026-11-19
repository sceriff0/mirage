# The samplesheet

<p class="standfirst">MIRAGE is driven by one CSV, passed with <code>--input</code>: one row per
image file, grouped by patient. This page is the full contract — the columns, the rules a sheet is
checked against at launch, the columns each entry point needs, and which image formats are read.</p>

!!! abstract "Canonical source"
    The required columns per entry point come from `lib/ParamUtils.groovy` (`STEPS`).

## A first samplesheet

```csv
patient_id,path_to_file,is_reference,channels
P001,/data/raw/P001_panel1.nd2,true,DAPI|CD3|CD8|CD4
P001,/data/raw/P001_panel2.nd2,false,DAPI|PANCK|SMA|VIMENTIN
P002,/data/raw/P002_panel1.czi,true,DAPI|CD3|CD8|CD4
P002,/data/raw/P002_panel2.czi,false,DAPI|FOXP3|KI67|CD20
```

Each row is one image (one acquisition panel or staining round). Rows are grouped by
`patient_id`, and each patient is processed independently and in parallel.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `patient_id` | string | Grouping key. All rows sharing it are registered into one coordinate space and produce one output tree. |
| `path_to_file` | path | The image. Any Bio-Formats-readable format (ND2, CZI, LIF, NDPI, TIFF, HDF5, OME-TIFF) — see [Supported formats](#supported-formats). |
| `is_reference` | `true` / `false` | Exactly **one** `true` per patient — the registration reference and the slide that gets segmented. |
| `channels` | `\|`-separated | Marker names in channel (acquisition) order, e.g. `DAPI\|PanCK\|CD45`. **Declared metadata, never parsed from the filename.** The count must match the image's channel count. |

## Rules

- **Exactly one reference per patient.** A patient with no `is_reference=true` row, or
  with two, is a **hard error at launch**, at every entry point. The pipeline never
  picks a reference for you: which slide the others are warped onto is your choice.
  Pick a round with a clean, complete nuclear stain.
- **Every row needs a nuclear marker.** `channels` must include one of
  `nuclear_markers` (default `DAPI`, `CELLTOX`).
- **A marker name may appear once per patient**, and once within a slide's own
  `channels` list. A nuclear marker is the exception and may repeat on every slide —
  it is the registration fiducial. Any other channel carried by two of a patient's
  slides is **refused at launch**, because only one copy survives per patient and
  which slide's copy that is would be decided by row order. Rename one of them
  (`PANCK_cycle1` / `PANCK_cycle2`) or, if it really is a nuclear stain, add it to
  `nuclear_markers`. Comparison is case-insensitive and ignores surrounding spaces.
  The check runs at every `--start`, checkpoint CSVs included; if a checkpoint trips
  it, re-run the earlier step from a corrected samplesheet rather than editing the
  checkpoint.

!!! warning "The nuclear channel is resolved by name, not position"
    `nuclear_markers` (default `['DAPI', 'CELLTOX']`) is an **ordered preference
    list**. `CONVERT_IMAGE` finds the first of those markers present in the
    row's `channels` and moves it to index 0. That one channel then drives
    **both** cell segmentation and the registration fiducial. If no listed
    marker is present the run fails fast at validation — single-channel images
    excepted. Matching is case-insensitive **substring**, so `DAPI_nuclear`
    counts as nuclear, and a DAPI-free row with `CELLTOX` is valid.

Validation runs before any task is submitted (and is exercised by a dry run, see
[Command line](usage.md#boolean-parameters)): per-row format, per-patient reference
counts, channel-count agreement, and file existence.

??? tip "Finding channel names from OME metadata"
    ```bash
    showinf -nopix -omexml-only /data/raw/P001_panel1.nd2 | grep -i "Channel"
    ```
    Or in Python: `AICSImage("file.nd2").channel_names`. Join the names with `|`.

## Columns per entry point { #columns-per-entry-point }

The required columns change with `--start`, because each step's entry point is the
previous step's checkpoint and consumes a different kind of image:

| `--start` | Image column | Other required columns | Typical source |
|---|---|---|---|
| `preprocessing` *(default)* | `path_to_file` | `patient_id`, `is_reference`, `channels` | your raw samplesheet |
| `registration` | `preprocessed_image` | `patient_id`, `is_reference`, `channels` | `<outdir>/csv/preprocessed.csv` |
| `segmentation` | `registered_image` | `patient_id`, `is_reference`, `channels` | `<outdir>/csv/registered.csv` |
| `postprocessing` | `registered_image` | `patient_id`, `is_reference`, `channels`, `cell_mask`, `nuclei_mask` | `<outdir>/csv/segmented.csv` |

You rarely write the later samplesheets by hand: each step writes the next step's
samplesheet as a checkpoint CSV — see [Checkpoints & resuming](resuming.md).

## Supported formats { #supported-formats }

Which formats this pipeline reads, and what each was verified against, is
recorded in [Format validation](validation/format_validation.md) — synthesised
fixtures for everything CI can generate (pyramidal OME-TIFF, BigTIFF, RGB,
8-bit, float32, HDF5, NDPI/NDPIS) on every push. The five vendor formats that
cannot be synthesised (`.czi`, `.nd2`, `.lif`, `.ndpi` real scanner bytes,
`.svs`) each need a run against real vendor files, and none has been recorded
yet — the page marks all five "kit-validated: pending" rather than omitting them.

## Other inputs

| Input | Parameter | Notes |
|---|---|---|
| StarDist model dir | `--segmentation_model_dir` | Optional. Download cache for the default Hugging Face model (`DSUnit/stardist`, pinned commit), or the folder of your own model. A model already there is used without network access. |
| InstanSeg cache | `--instanseg_model_dir` | Writable BioImage.IO cache; exported as `INSTANSEG_BIOIMAGEIO_PATH`. |
| CellSAM weights | `--cellsam_model_path` | Pre-downloaded weights. If unset, weights auto-download and need `DEEPCELL_ACCESS_TOKEN` in the launch environment. |
