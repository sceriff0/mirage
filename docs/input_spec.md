# Input Format

MIRAGE uses CSV manifests. Required columns depend on `--step`.

## `--step preprocessing`

Required columns:

- `patient_id`
- `path_to_file`
- `is_reference`
- `channels`

Example:

```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_panel1.nd2,true,DAPI|CD3|CD8|CD4
P001,/data/P001_panel2.nd2,false,DAPI|PANCK|SMA|VIMENTIN
```

## `--step registration`

Required columns:

- `patient_id`
- `preprocessed_image`
- `is_reference`
- `channels`

Example:

```csv
patient_id,preprocessed_image,is_reference,channels
P001,/results/P001/preprocessed/P001_corrected.ome.tif,true,DAPI|CD3|CD8|CD4
P001,/results/P001/preprocessed/P001_panel2_corrected.ome.tif,false,DAPI|PANCK|SMA|VIMENTIN
```

## `--step postprocessing`

Required columns:

- `patient_id`
- `registered_image`
- `is_reference`
- `channels`

Example:

```csv
patient_id,registered_image,is_reference,channels
P001,/results/P001/registered/P001_registered.ome.tiff,true,DAPI|CD3|CD8|CD4
P001,/results/P001/registered/P001_panel2_registered.ome.tiff,false,DAPI|PANCK|SMA|VIMENTIN
```

## `--step copy_results`

No CSV is required.

Required runtime args:

- `--outdir`
- `--savedir`

## Validation Rules

- Exactly one reference image per patient at preprocessing input.
- Input files must exist.
- `channels` must include DAPI and match expected image content/ordering requirements.
- For checkpoint steps, required columns must be present for that step.

