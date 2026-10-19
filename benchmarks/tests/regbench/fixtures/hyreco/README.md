# HyReCo fixture (synthetic)

Everything here is SYNTHETIC: smooth random blobs written by
`benchmarks/tests/regbench/make_landmark_fixtures.py`. No pixel and no landmark of HyReCo
is in it. Only the LAYOUT and the FORMATS are the dataset's. Each pair's PHH3 (moving)
image is its HE (reference) image shifted by `SHIFT_PX = (4, -3)` pixels, and its
landmarks are the HE landmarks plus that shift.

Regenerate from the repository root:

    python -m benchmarks.tests.regbench.make_landmark_fixtures

## Layout and formats, and where each comes from

| detail | here | source |
|---|---|---|
| subset A in `HyReCo/<stain>/<id>.tif` + `.csv`, subset B in `HyReCo-Additional/<stain>/<id>.tif` + `.csv` | `HyReCo/{HE,PHH3}/{29,533}`, `HyReCo-Additional/{HE,PHH3}/{8,29}` | `5a46f778:benchmarks/regbench/datasets/hyreco.py:50-55` |
| one CSV per image: lines `x,y,z` in millimetres, no header, origin the upper-left corner of the image | `x,y,0` in mm, no header | `5a46f778:benchmarks/regbench/datasets/hyreco.py:57-59` |
| pixel size 0.24 um | 0.24 um, used for the mm conversion | `5a46f778:benchmarks/regbench/datasets/hyreco.py:60-63` |
| the TIFF states its pixel size in its resolution tags (inch or centimetre) or as `MPP = ` in the image description; with neither, 0.24 is assumed and said | `29`: resolution tags in cm; `533`: `MPP = 0.24`; `b/8`: neither | `5a46f778:benchmarks/regbench/datasets/hyreco.py:78-82`, `:274-300` (`pixel_size`) |
| the first series' first level is the full-resolution RGB uint8 image | one level, 96 x 128 RGB, 32 px tiles, zlib | `5a46f778:benchmarks/regbench/datasets/hyreco.py:67-68` |
| the same id in both subsets (here `29`) | `a_29` and `b_29`, flagged `shared_id` | ruling 2026-10-19 (hyreco); the id lists are on the dataset page cited at `5a46f778:benchmarks/regbench/datasets/hyreco.py:50-57` |

The source module's own tags are `[FULL: page; the archives were never opened]` for the
layout and `[PARTIAL: ...]` for the paper (see the cited lines). The real files were
never opened, so every format detail above rests on the dataset page's text, not on a
file.
