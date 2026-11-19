"""Oracle test for the vectorized get_matched_masks refactor.

The CSE golden fixture only exercises the trivial matching path. This test keeps
the per-cell set-of-tuples algorithm as an oracle and asserts the vectorized
version produces byte-identical matched masks on scenes that hit the
branches that matter: mismatch-repair, a nucleus spanning two cells' interiors,
a nucleus overlapping only a cell's membrane (interior overlap 0 -> unmatched),
multi-candidate min-mismatch selection with greedy claiming, and skipped cells.
Plus randomized fuzz scenes for breadth.
"""

import numpy as np
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries

from bin.utils.cse.functions import get_matched_masks


# ---------------------------------------------------------------------------
# Frozen oracle: the per-cell algorithm, helpers inlined, as the behaviour contract.
# ---------------------------------------------------------------------------
def _old_get_indices_pandas(data):
    import pandas as pd

    d = data.ravel()

    def f(x):
        return np.unravel_index(x.index, data.shape)

    return pd.Series(d).groupby(d).apply(f)


def _old_compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix(
        (cols, (data.ravel(), cols)), shape=(np.int64(data.max() + 1), data.size)
    )


def _old_get_indices_sparse(data):
    data = data.astype(np.uint64)
    M = _old_compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def _old_get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def _old_get_boundary(mask):
    mask_boundary = find_boundaries(mask, mode="inner")
    return _old_get_indexed_mask(mask, mask_boundary)


def _old_get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair):
    a = set((tuple(i) for i in cell_arr))
    b = set((tuple(i) for i in cell_membrane_arr))
    c = set((tuple(i) for i in nuclear_arr))
    d = a - b
    mismatch_pixel_num = len(list(c - d))
    mismatch_fraction = len(list(c - d)) / len(list(c))
    if not mismatch_repair:
        if mismatch_pixel_num == 0:
            return np.array(list(a)), np.array(list(c)), 0
        return False, False, False
    if mismatch_pixel_num < len(c):
        return np.array(list(a)), np.array(list(d & c)), mismatch_fraction
    return False, False, False


def _old_get_mask(cell_list, mask_shape):
    mask = np.zeros(mask_shape)
    for cell_num in range(len(cell_list)):
        mask[tuple(cell_list[cell_num].T)] = cell_num + 1
    return mask


def _old_get_matched_masks(cell_mask, nuclear_mask):
    cell_membrane_mask = _old_get_boundary(cell_mask)
    cell_coords = _old_get_indices_pandas(cell_mask)[1:]
    nucleus_coords = _old_get_indices_pandas(nuclear_mask)[1:]
    cell_membrane_coords = _old_get_indices_sparse(cell_membrane_mask)[1:]
    cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
    cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))
    nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
    cell_matched_index_list = []
    nucleus_matched_index_list = []
    cell_matched_list = []
    nucleus_matched_list = []
    for i in range(len(cell_coords)):
        if len(cell_coords[i]) != 0:
            current_cell_coords = cell_coords[i]
            nuclear_search_num = np.unique(
                nuclear_mask[current_cell_coords[:, 0], current_cell_coords[:, 1]]
            )
            best_mismatch_fraction = 1
            whole_cell_best = []
            for j in nuclear_search_num:
                if j != 0:
                    if (j - 1 not in nucleus_matched_index_list) and (
                        i not in cell_matched_index_list
                    ):
                        whole_cell, nucleus, mismatch_fraction = _old_get_matched_cells(
                            cell_coords[i],
                            cell_membrane_coords[i],
                            nucleus_coords[j - 1],
                            mismatch_repair=1,
                        )
                        if type(whole_cell) is not bool:
                            if mismatch_fraction < best_mismatch_fraction:
                                best_mismatch_fraction = mismatch_fraction
                                whole_cell_best = whole_cell
                                nucleus_best = nucleus
                                i_ind = i
                                j_ind = j - 1
            if len(whole_cell_best) > 0:
                cell_matched_list.append(whole_cell_best)
                nucleus_matched_list.append(nucleus_best)
                cell_matched_index_list.append(i_ind)
                nucleus_matched_index_list.append(j_ind)
    cell_matched_mask = _old_get_mask(cell_matched_list, cell_mask.shape)
    nuclear_matched_mask = _old_get_mask(nucleus_matched_list, nuclear_mask.shape)
    cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask
    return cell_matched_mask, nuclear_matched_mask, cell_outside_nucleus_mask


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------
def _relabel_contiguous(mask):
    uniq = np.unique(mask)
    lut = np.zeros(int(uniq.max()) + 1, dtype=np.int32)
    pos = uniq[uniq != 0]
    lut[pos] = np.arange(1, len(pos) + 1, dtype=np.int32)
    return lut[mask]


def _handcrafted_scene():
    """A 64x64 scene hitting every matching branch."""
    cell = np.zeros((64, 64), np.int32)
    nuc = np.zeros((64, 64), np.int32)

    # A: clean centered nucleus (mismatch 0)
    cell[2:12, 2:12] = 1
    nuc[5:9, 5:9] = 1

    # B: two candidate nuclei -> must pick the fully-interior one (min mismatch),
    #    and one nucleus pokes outside the cell (repair path).
    cell[2:12, 20:30] = 2
    nuc[4:8, 26:34] = 2  # pokes out to the right (cols 30-33 outside cell B)
    nuc[6:9, 22:25] = 3  # fully interior -> lower mismatch, should win

    # C and D adjacent; one nucleus spans both interiors -> greedy claims for C,
    #    D has no other nucleus -> skipped.
    cell[20:30, 2:12] = 4
    cell[20:30, 12:22] = 5
    nuc[23:27, 9:15] = 4  # straddles the C|D border

    # E: no nucleus at all -> skipped.
    cell[40:50, 2:12] = 6

    # F: nucleus overlaps only the cell's top membrane row (interior overlap 0).
    cell[40:50, 20:30] = 7
    nuc[40:41, 22:28] = 5  # row 40 is F's inner boundary -> unmatched

    return _relabel_contiguous(cell), _relabel_contiguous(nuc)


def _random_scene(seed, size=80, n_cells=12):
    rng = np.random.default_rng(seed)
    cell = np.zeros((size, size), np.int32)
    nuc = np.zeros((size, size), np.int32)
    cid = 0
    # non-overlapping cell tiles on a coarse grid, random nucleus inside/near each
    step = size // 4
    for gy in range(1, size - step, step):
        for gx in range(1, size - step, step):
            if rng.random() < 0.2:
                continue  # some cells missing -> gaps handled by relabel
            cid += 1
            h = rng.integers(6, step - 1)
            w = rng.integers(6, step - 1)
            cell[gy : gy + h, gx : gx + w] = cid
            if rng.random() < 0.85:
                # nucleus offset by a random amount -> sometimes pokes outside
                oy = rng.integers(-2, 4)
                ox = rng.integers(-2, 4)
                nh = rng.integers(2, max(3, h - 2))
                nw = rng.integers(2, max(3, w - 2))
                y0 = max(0, gy + 2 + oy)
                x0 = max(0, gx + 2 + ox)
                nuc[y0 : y0 + nh, x0 : x0 + nw] = cid
    return _relabel_contiguous(cell), _relabel_contiguous(nuc)


def _assert_same(cell, nuc):
    o = _old_get_matched_masks(cell, nuc)
    n = get_matched_masks(cell, nuc)
    names = ("cell_matched", "nuclear_matched", "cell_outside_nucleus")
    for name, ov, nv in zip(names, o, n):
        assert np.array_equal(ov.astype(np.int64), nv.astype(np.int64)), (
            f"{name} differs: old-uniq={np.unique(ov)} new-uniq={np.unique(nv)}"
        )


def test_handcrafted_scene_matches_oracle():
    cell, nuc = _handcrafted_scene()
    _assert_same(cell, nuc)


def test_random_scenes_match_oracle():
    for seed in range(20):
        cell, nuc = _random_scene(seed)
        _assert_same(cell, nuc)
