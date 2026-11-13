# mirage: Citations

## Core tools

The MIRAGE pipeline builds on the following tools and methods. They are grouped
by the stage of the pipeline in which they are used. Please cite the ones that
apply to the steps you ran (in particular, the segmentation backend you selected
via `--seg_method`).

### Image I/O

#### Bio-Formats — Image reading / metadata access

> Linkert, M., Rueden, C. T., Allan, C., Burel, J.-M., Moore, W., Patterson, A., Loranger, B., Moore, J., Neves, C., MacDonald, D., Tarkowska, A., Sticco, C., Hill, E., Rossner, M., Eliceiri, K. W., & Swedlow, J. R. (2010).
> Metadata matters: access to image data in the real world.
> *Journal of Cell Biology*, 189(5), 777–782.
> [https://doi.org/10.1083/jcb.201004104](https://doi.org/10.1083/jcb.201004104)

GitHub: <https://github.com/ome/bioformats>

#### BioIO — Image reading

`CONVERT_IMAGE` reads every input through BioIO (bioio 3.5.0), which uses Bio-Formats for
formats without a native BioIO reader. The MIRAGE paper cites no separate reference for
BioIO; cite Bio-Formats (above) and name the bioio version you ran.

---

### Illumination correction

#### BaSiCPy / BaSiC — Background and shading correction

> Peng, T., Thorn, K., Schroeder, T., Wang, L., Theis, F. J., Marr, C., & Navab, N. (2017).
> A BaSiC tool for background and shading correction of optical microscopy images.
> *Nature Communications*, 8, 14836.
> [https://doi.org/10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)

GitHub: <https://github.com/peng-lab/BaSiCPy>

---

### Registration

#### STARE — Tile-based non-rigid registration (`registration_method = tiled`, the default)

> Fassi, V., Borriero, C., Dondi, A., Chiodin, A., Compagnoni, M., Frigè, G., Chiocca, S., Bossi, P., Mazzarella, L., Barberis, M., Rodighiero, S., & Zhan, Y.
> STARE enables whole-slide multiplexed immunofluorescence on conventional microscopes by bounded-memory non-rigid registration.
> Manuscript (no DOI yet).

GitHub: <https://github.com/sceriff0/stare> (the `stare-registration` package; MIRAGE pins v1.3.0)

STARE's slide-wide solve smooths the displacement vectors by robust penalized least squares
with the discrete cosine transform (DCT-PLS):

> Garcia, D. (2010).
> Robust smoothing of gridded data in one and higher dimensions with missing values.
> *Comput. Stat. Data Anal.*, 54, 1167–1178.
> [https://doi.org/10.1016/j.csda.2009.09.020](https://doi.org/10.1016/j.csda.2009.09.020)

---

#### VALIS — Whole-slide image registration

> Gatenbee, C. D. et al. (2023).
> Virtual alignment of pathology image series for multi-gigapixel whole slide images.
> *Nature Communications*, 14, 4502.
> [https://doi.org/10.1038/s41467-023-40218-9](https://doi.org/10.1038/s41467-023-40218-9)

GitHub: <https://github.com/MathOnco/valis>

---

### Segmentation

MIRAGE supports three interchangeable segmentation backends selected via
`--seg_method`. Cite the backend(s) you actually ran.

#### StarDist — Star-convex nuclei / cell segmentation

> Schmidt, U., Weigert, M., Broaddus, C., & Myers, G. (2018).
> Cell Detection with Star-Convex Polygons.
> *Medical Image Computing and Computer Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science, vol. 11071.
> [https://doi.org/10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30)

GitHub: <https://github.com/stardist/stardist>

---

#### InstanSeg — Embedding-based instance segmentation

> Goldsborough, T., O'Callaghan, A., Inglis, F., Leplat, L., Filby, A., Bilen, H., & Bankhead, P. (2024).
> InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation.
> *arXiv*:2408.15954.
> [https://arxiv.org/abs/2408.15954](https://arxiv.org/abs/2408.15954)

For the channel-invariant fluorescence model used on multiplexed images, also cite:

> Goldsborough, T., O'Callaghan, A., Inglis, F., Visvanathan, A., Bilen, H., & Bankhead, P. (2024).
> A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg.
> *bioRxiv*.
> [https://doi.org/10.1101/2024.09.04.611150](https://doi.org/10.1101/2024.09.04.611150)

GitHub: <https://github.com/instanseg/instanseg>

---

#### CellSAM — Foundation-model cell segmentation

> Marks, M. et al. (2025).
> CellSAM: a foundation model for cell segmentation.
> *Nature Methods*, 22, 2585–2593.
> [https://doi.org/10.1038/s41592-025-02879-w](https://doi.org/10.1038/s41592-025-02879-w)

Preprint:

> Israel, U., Marks, M., Dilip, R., Li, Q., Yu, C., Laubscher, E., Iqbal, A., Pradhan, E., Ates, A., Abt, M., Brown, C., Pao, E., Restrepo, S., Van Valen, D., et al. (2023).
> A Foundation Model for Cell Segmentation.
> *bioRxiv*.
> [https://doi.org/10.1101/2023.11.17.567630](https://doi.org/10.1101/2023.11.17.567630)

GitHub: <https://github.com/vanvalenlab/cellSAM>

CellSAM builds on the Segment Anything Model (SAM):

> Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023).
> Segment Anything.
> *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*; *arXiv*:2304.02643.
> [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)

---

### Quantification & morphology

#### scikit-image — Morphology, label expansion, contours

> van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., & the scikit-image contributors (2014).
> scikit-image: image processing in Python.
> *PeerJ*, 2, e453.
> [https://doi.org/10.7717/peerj.453](https://doi.org/10.7717/peerj.453)

GitHub: <https://github.com/scikit-image/scikit-image>

---

### Downstream analysis

#### QuPath — Digital pathology analysis (GeoJSON target)

> Bankhead, P., Loughrey, M. B., Fernández, J. A., Dombrowski, Y., McArt, D. G., Dunne, P. D., McQuaid, S., Gray, R. T., Murray, L. J., Coleman, H. G., James, J. A., Salto-Tellez, M., & Hamilton, P. W. (2017).
> QuPath: Open source software for digital pathology image analysis.
> *Scientific Reports*, 7, 16878.
> [https://doi.org/10.1038/s41598-017-17204-5](https://doi.org/10.1038/s41598-017-17204-5)

Website: <https://qupath.github.io/>

#### SpatialData — the `.zarr` store written by `EXPORT_SPATIALDATA`

> Marconato, L. et al. (2025).
> SpatialData: an open and universal data framework for spatial omics.
> *Nature Methods*, 22, 58–62.
> [https://doi.org/10.1038/s41592-024-02212-x](https://doi.org/10.1038/s41592-024-02212-x)

#### scverse — the ecosystem that reads the store without conversion

> Virshup, I. et al. (2023).
> The scverse project provides a computational ecosystem for single-cell omics data analysis.
> *Nature Biotechnology*, 41, 604–606.
> [https://doi.org/10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8)

---

### Workflow framework

#### Nextflow — Workflow management

> Di Tommaso, P., Chatzou, M., Floden, E. W., Barja, P. P., Palumbo, E., & Notredame, C. (2017).
> Nextflow enables reproducible computational workflows.
> *Nature Biotechnology*, 35, 316–319.
> [https://doi.org/10.1038/nbt.3820](https://doi.org/10.1038/nbt.3820)

#### nf-core — Pipeline framework and community

> Ewels, P. A., Peltzer, A., Fillinger, S., Patel, H., Alneberg, J., Wilm, A., Garcia, M. U., Di Tommaso, P., & Nahnsen, S. (2020).
> The nf-core framework for community-curated bioinformatics pipelines.
> *Nature Biotechnology*, 38, 276–278.
> [https://doi.org/10.1038/s41587-020-0439-x](https://doi.org/10.1038/s41587-020-0439-x)

---

## Related / legacy

### Pixie / ark-analysis — Pixel and cell clustering

Pixie pixel/cell clustering is **not currently wired into the MIRAGE pipeline**.
It is kept here for reference only (it lives in the legacy code paths) and is not
a current runtime dependency.

> Angelo Lab. ark-analysis: Pixie multiplexed imaging analysis pipeline.

GitHub: <https://github.com/angelolab/pixie>
