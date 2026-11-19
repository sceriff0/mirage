# GPU or no GPU

<p class="standfirst">Only segmentation can use a GPU. Registration with STARE never does. With no
NVIDIA GPU, add one preset file to the command and segmentation runs on the processor: slower, same
result.</p>

## Do I need a GPU?

No. Segmentation uses a GPU by default (`seg_gpu = true`); everything else runs on the
CPU. To run segmentation on the CPU too, add the preset:

```bash
nextflow run . -profile local,docker \
  -params-file params/no_gpu.json \
  --input samplesheet.csv --outdir results
```

`params/no_gpu.json` carries exactly one setting, `"seg_gpu": false`. It is a boolean,
so it goes in a `-params-file`, never on the command line (see
[Command line → Boolean parameters](usage.md#boolean-parameters)). The `test` profile
sets it too, so the demo runs on any laptop.

Use it on any Mac, on most laptops, and on any machine or cluster node without an NVIDIA
GPU. When in doubt on a new machine, start with `seg_gpu = false`.

!!! warning
    If you leave `seg_gpu = true` without exposing a GPU to the container, segmentation
    will fail or silently fall back to the CPU, depending on the backend.

## What `seg_gpu = true` adds

`SEGMENT` requests a GPU when `seg_gpu = true`, and so does `SEG_QC_SEGMENT` — it is the
same process under an alias, and Nextflow matches `withName: 'SEGMENT'` against an
alias' original name. `SEG_QC_GEOJSON` does not: it only traces contours, which is pure
CPU.

| Engine | What is added |
|---|---|
| SLURM | `clusterOptions` composes `--gres=gpu:${params.gpu_type}` (default `1`, i.e. any one GPU; set a typed string in your site config) with the `slurm` profile's `--account`/`--qos`, inlined per-block since a `withName:` assignment replaces rather than merges with the profile's own `clusterOptions` — see `conf/modules.config`'s `SEGMENT` block |
| Docker | `containerOptions = --gpus all` |
| Singularity | `containerOptions = --nv` |

Set `seg_gpu = false` to force CPU; the `--gres` request and the container flags are
both dropped.

## On a cluster

- Match `--gpu_type` to your cluster's GRES string (`sinfo -o "%G"`). A bare `1` (the
  default) asks for any one GPU; a typed string such as `nvidia_a100:1` goes in your
  `site.config`. The request is emitted as `--gres=gpu:<value>`.
- Without `--nv`, Singularity does not bind the host NVIDIA driver stack and
  `torch.cuda.is_available()` is `False` even when SLURM has granted the GPU — CellSAM
  then silently falls back to CPU, which turns a whole-slide run into a multi-day job.
  The `singularity` profile adds `--nv` for you whenever `seg_gpu = true`.

## Backend notes

- **InstanSeg**: `seg_instantseg_batch_size` (default 16) is the number of tiles per
  forward pass. Raise it on a big GPU; lower it to avoid running out of GPU memory.
- **CellSAM** on the CPU is very slow on whole slides (see above).
