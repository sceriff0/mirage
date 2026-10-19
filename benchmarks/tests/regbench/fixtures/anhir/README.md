# ANHIR fixture (synthetic)

Everything here is SYNTHETIC: smooth random blobs written by
`benchmarks/tests/regbench/make_landmark_fixtures.py`. No pixel and no landmark row of
ANHIR is in it (ANHIR is CC BY-NC-SA). Only the LAYOUT and the FORMATS are the dataset's.
Each pair's moving (source) image is its target image shifted by `SHIFT_PX = (4, -3)`
pixels, and its source landmarks are the target landmarks plus that shift. So a method
that recovers the shift scores (almost) zero rTRE.

Regenerate from the repository root:

    python -m benchmarks.tests.regbench.make_landmark_fixtures

## Layout and formats, and where each comes from

| detail | here | source |
|---|---|---|
| the source folder holds the cover table `dataset_medium.csv`, the image archive unpacked under `images/` and the landmark archive unpacked under `landmarks/`, each with the table's relative paths | `dataset_medium.csv`, `images/`, `landmarks/` | `5a46f778:benchmarks/regbench/datasets/anhir.py:60-63` |
| the image archive is split (`dataset_medium.z01..` + `.zip`), rejoined with `zip -s 0` and unzipped into `images/`; the landmark archive is a separate download with the same tree, unpacked into `landmarks/` | (the unpacked result only) | `5a46f778:benchmarks/anhir/prepare.py:1-25, 75-92` |
| cover-table columns `Source image`, `Target image`, `Source landmarks`, `Target landmarks`, `status`, `Image diagonal [pixels]`, plus an unnamed index column | the 11-column header, copied from the local copy's header line | `5a46f778:benchmarks/regbench/datasets/anhir.py:85-88`, `:151-175` (`read_table`); the header line of the inspected local copy |
| roles: `Source image` = moving, `Target image` = reference, landmarks paired by row | source = target + shift | `5a46f778:benchmarks/regbench/datasets/anhir.py:30-35` |
| landmark CSV: header `,X,Y`, an index column, pixel coordinates at the folder's scale, origin the top-left pixel | `,X,Y` then `i,x,y` | `5a46f778:benchmarks/regbench/datasets/anhir.py:37-41`, `:296-320` (`read_landmarks`); first two lines of one landmark file of the local copy |
| set folders `<set>/scale-<N>pc/<stem>.<ext>`: JPEG, PNG for the lung lobes | `COAD_01/scale-25pc/*.jpg`, `COAD_02/scale-25pc/*.jpg`, `lung-lobes_1/scale-100pc/*.png` | `5a46f778:benchmarks/regbench/datasets/anhir.py:44, 52-53, 144-147` |
| `Image size [pixels]` written `(W, H)`, `Image diagonal [pixels]` one decimal | `(80, 60)`, `100.0` | the first data rows of the inspected local copy |
| `status` is `training` or `evaluation`; an evaluation pair has no public target landmarks | row 0 is `evaluation`, with no target CSV | `5a46f778:benchmarks/regbench/datasets/anhir.py:89-90` (`TRAINING`, `N_TRAINING`) |

The "inspected local copy" is the real download at `challenge/anhir/` in the main
checkout, which was read on 2026-10-19 only for the header line, the first data rows and
the first two lines of one landmark file. Nothing from it was copied here but the column
names and the number formats.

The table indices (0, 4, 12, 370), the set names and the stems are ANHIR-shaped
identifiers chosen for the fixture. They are not checked against the real table's rows.
