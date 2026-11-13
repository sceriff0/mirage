# Citation

If you use MIRAGE in your research, please cite the pipeline itself **and** the underlying tools your run used.

## Citing MIRAGE

MIRAGE, STARE and FlowPath are described together in:

> V. Fassi, C. Borriero, A. Dondi, A. Chiodin, M. Compagnoni, G. Frigè, S. Chiocca, P. Bossi,
> L. Mazzarella, M. Barberis, S. Rodighiero and Y. Zhan. *STARE enables whole-slide multiplexed
> immunofluorescence on conventional microscopes by bounded-memory non-rigid registration.*
> Manuscript in preparation.

Until it is published, cite the software. Include in your methods section:

- The pipeline name: **MIRAGE**
- The version: a tagged release (e.g. `v1.0.0`) or the commit SHA from `git rev-parse HEAD`
- The repository URL: <https://github.com/sceriff0/mirage>
- The registration mode you ran. For STARE, also its version, recorded in
  [`requirements/stare.txt`](https://github.com/sceriff0/mirage/blob/main/requirements/stare.txt)
  (`stare-registration` 1.3.0), and its repository, <https://github.com/sceriff0/stare>

No DOI is registered for the software yet. When one is, it will appear in
[`CITATION.cff`](https://github.com/sceriff0/mirage/blob/main/CITATION.cff) at the repository
root and on the README badge. If you ran an untagged commit, cite the commit SHA — that is the
only thing that makes an untagged run reproducible.

[The paper](paper.md) states what the manuscript reports and where each part lives in the software.

## Tool citations

The full list of underlying-tool citations is maintained at the repository root as [`CITATIONS.md`](https://github.com/sceriff0/mirage/blob/main/CITATIONS.md) and is included verbatim below.

!!! tip "Cite the backend you ran"
    MIRAGE ships three interchangeable segmentation backends — **StarDist**, **InstanSeg**, and **CellSAM** — selected at runtime via `--seg_method`. Cite the segmentation reference(s) that correspond to the backend(s) you actually used.

{% include-markdown "../CITATIONS.md" heading-offset=2 %}
