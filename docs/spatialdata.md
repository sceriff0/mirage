# SpatialData export & FlowPath join

MIRAGE writes a scverse-native [SpatialData](https://spatialdata.scverse.org/)
`.zarr` alongside the pyramidal OME-TIFF and the QuPath GeoJSON. The TIFF and
GeoJSON remain the primary outputs — the `.zarr` is additive, and exists so the
same run can be opened by `scanpy`, `squidpy`, `napari-spatialdata`, LazySlide
and Vitessce without conversion.

Nothing is recomputed. Four of SpatialData's five element types are
re-serializations of artifacts postprocessing already produced.

## What the store contains

```
<outdir>/<patient>/spatialdata/<patient>.zarr
  labels/cell_mask          # from SEGMENT
  labels/nuclei_mask
  shapes/cells              # polygons from EXTRACT_CELL_PROPERTIES
  shapes/nuclei
  tables/table              # AnnData
  images/pyramid            # only with --spatialdata_include_image
```

`points/` is absent by design: this is protein imaging, so there are no
transcripts to store.

### The table

| Slot | Contents |
|---|---|
| `X` | raw intensities; columns are the `"CD3: Nucleus: Median"` keys FlowPath also uses |
| `var` | parsed `marker` / `compartment` / `statistic` |
| `layers["zscore"]` | z-scores, computed column-wise across all cells of the patient |
| `obs` | `label`, `patient_id`, `fov`, morphology (`qc_area`, `qc_solidity`, …) |
| `obsm["spatial"]` | centroids, **in pixels**, corner-of-pixel (QuPath/ImageJ) — the same convention as the `shapes` polygons, so a cell's point lies inside its own polygon |
| `obsm["qc_reg_residual_px"]` | per-cell registration residual, one column per cycle |
| `uns["qc"]` / `uns["qc_json"]` | registration + segmentation QC, flattened and verbatim |
| `uns["provenance"]` | pipeline version, params, `versions.yml` |

Region metadata follows the SpatialData contract: `region="cell_mask"`,
`region_key="region"` (categorical), `instance_key="label"`.

!!! note "Coordinates are pixels, not micrometres"
    Elements are stored in intrinsic pixel coordinates and carry two coordinate
    systems: `global` (identity, pixels) and `um` (scaled by `pixel_size`).
    Storing µm directly would double-apply the scale on any cross-modality
    alignment.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `skip_spatialdata_export` | `false` | Skip the export entirely. |
| `spatialdata_include_image` | `false` | Write the pyramid into the store. `spatialdata.write()` materializes every element, so this duplicates the largest artifact the pipeline makes — on for a self-contained, depositable object; off for everyday runs. |
| `spatialdata_residual_join_max_px` | `15.0` | Max centroid distance when joining registration QC residuals onto `cell_mask`. |

### Per-cell registration confidence

`obsm["qc_reg_residual_px"]` is a cells × cycles matrix of centroid displacement
after registration, letting you drop cells in poorly-registered regions *before*
phenotyping — the cyclic-IF failure mode around tissue folds and slide edges.

It is a **spatial** join, not an identity join: `SEG_QC_SEGMENT` segments each
slide's native, pre-registration image, while `SEGMENT` segments the registered
reference, so the two share a coordinate frame but no label space. (They are the
same process under two names, so `--seg_method` picks the segmenter for both — but
different inputs mean different cells regardless.) Consequences:

* A cell with no QC evidence gets `NaN` and `obs["qc_reg_matched"] = False`.
  **Unmatched is not the same as well-registered.**
* Where several QC cells land on one mask cell, the **worst** residual wins —
  this column exists to exclude bad cells.
* Join quality per slide is recorded in `uns["qc_reg_residual_join"]`.

## Joining FlowPath phenotypes

Gating happens interactively in QuPath *after* the pipeline finishes, so it is
never part of the DAG. `bin/join_flowpath.py` folds the results back in and
builds a cohort-level dataset.

```bash
# one patient
join_flowpath.py \
    --zarr results/P001/spatialdata/P001.zarr \
    --flowpath P001_phenotypes.csv \
    --out-h5ad cohort.h5ad \
    --out-table cohort.parquet

# a whole cohort, also writing phenotypes back into each store
join_flowpath.py \
    --zarr results/*/spatialdata/*.zarr \
    --flowpath phenotypes/*.csv \
    --gate-tree gates.json \
    --out-h5ad cohort.h5ad \
    --out-table cohort.parquet \
    --out-stats join_stats.json \
    --update-zarr
```

Run it in the export image:

```bash
docker run --rm -v "$PWD":/data -w /data \
    bolt3x/mirage-spatialdata:1.0.0 \
    /data/bin/join_flowpath.py --zarr ... --flowpath ... --out-h5ad ...
```

### What it adds

| Slot | Contents |
|---|---|
| `obs["phenotype"]` | categorical, from the gate-tree leaf; unmatched cells become `unclassified` |
| `obsm["positivity"]` | boolean cells × **gated** columns |
| `uns["positivity_columns"]` | the measurement each positivity column refers to |
| `obs["fp_outlier"]`, `obs["fp_out_of_annotation"]` | FlowPath exclusion flags |
| `obs["fp_matched"]` | whether this cell was found in the CSV at all |
| `uns["flowpath"]["gate_tree"]` | the serialized tree, as provenance |

!!! warning "Cells are never matched positionally"
    FlowPath's `cell_id` is a row index into a QuPath `Collection<PathObject>`
    whose ordering is not guaranteed. Aligning on it shifts phenotype
    assignments the moment a cell is deleted or a filter drops rows — producing
    plausible-looking wrong biology rather than an error.

    The script joins on `label` when FlowPath exports it, and otherwise falls
    back to a **mutual-nearest centroid** join. That fallback is exact because
    FlowPath re-exports the `Centroid X µm` measurement MIRAGE itself wrote, and
    both it and `obsm["spatial"]` are corner-of-pixel, so `x_px = x_um /
    pixel_size` inverts it cleanly — no half-pixel correction. If fewer than
    `--min-match-fraction` (default 50%) of cells match, the script fails rather
    than writing a mostly-unlabelled dataset.

### Positivity is three-state

FlowPath's `<column>_sign` is `"+"`, not-`"+"`, or **blank when no gate anywhere
in the tree touched that column**. Only gated columns are carried into
`obsm["positivity"]`, so an absent column means "never gated". A boolean cast
over every marker would silently merge that with "negative".

## Reading it back

!!! info "`layers` contains a `None` key"
    On read-back, `adata.layers` lists `['zscore', None]`. This is an **anndata
    0.13.2 zarr round-trip artifact, not a MIRAGE bug** — a plain AnnData with no
    SpatialData involvement gains the same key, and `layers[None]` is simply `X`.
    Ignore it; `adata.layers["zscore"]` works normally.

```python
import spatialdata as sd
import scanpy as sc

sdata = sd.read_zarr("results/P001/spatialdata/P001.zarr")
adata = sdata.tables["table"]

# nuclear measurements only
nuclear = adata[:, adata.var.compartment == "Nucleus"]

# drop cells with poor registration in any cycle
import numpy as np

keep = adata.obs["qc_reg_matched"] & (adata.obs["qc_reg_residual_max_px"] < 5)

# the segmentation quality score, with the factor it depends on
qc = adata.uns["qc"]["segmentation"]
```
