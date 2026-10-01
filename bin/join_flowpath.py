#!/usr/bin/env python3
"""Join MIRAGE SpatialData stores with FlowPath phenotype exports.

Takes one or more per-patient ``.zarr`` stores written by ``EXPORT_SPATIALDATA``
and the matching FlowPath ``PhenotypeCsvExporter`` CSVs, and produces a single
analysis-ready cohort dataset carrying marker intensities, morphology, pipeline
QC, **per-marker positivity**, and **phenotype** calls.

Outputs (any combination):

* ``--out-h5ad``   one concatenated AnnData across every patient — the "full dataset"
* ``--out-table``  the same thing flattened to CSV/Parquet, one row per cell
* ``--update-zarr`` writes the phenotype columns back into each source store

Joining
-------
Cells are matched on the segmentation ``label`` when FlowPath exports it, and
otherwise on **centroid position**. They are never matched positionally.

FlowPath's ``cell_id`` is the row index into a QuPath ``Collection<PathObject>``
whose ordering is not guaranteed, so aligning on it silently shifts phenotype
assignments the moment a cell is deleted, a quality filter drops rows, or QuPath
returns detections in a different order — which produces plausible-looking wrong
biology rather than an error. The centroid fallback is safe because FlowPath
exports ``centroid_x``/``centroid_y`` in micrometres, taken from the
``Centroid X µm`` measurement MIRAGE itself wrote, so the mapping back to pixels
is exact: ``x_px = x_um / pixel_size``. Both sides of that comparison are
corner-of-pixel and converted by the same rule
(``bin/utils/pixel_convention.py``), so nothing is left to correct here.

Positivity
----------
FlowPath's ``<column>_sign`` is three-state: ``"+"``, not-``"+"``, or blank when
no gate anywhere in the tree touched that column. Only gated columns are carried
into ``obsm['positivity']``, so an absent column means "never gated" — a boolean
cast over every marker would merge that with "negative".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from logger import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

INSTANCE_KEY = "label"
SIGN_SUFFIX = "_sign"
ZSCORE_SUFFIX = "_zscore"

# Columns PhenotypeCsvExporter writes before the per-marker triplets.


# ── FlowPath CSV parsing ───────────────────────────────────────────────────────
def gated_sign_columns(df: pd.DataFrame) -> List[str]:
    """``*_sign`` columns that at least one cell was actually gated on.

    A column of entirely blank signs means no gate in the tree touched it; keeping
    it would assert "all negative" where the truth is "never measured against a
    threshold".
    """
    out = []
    for c in df.columns:
        if not c.endswith(SIGN_SUFFIX):
            continue
        vals = df[c].astype("string").fillna("")
        if (vals.str.strip() != "").any():
            out.append(c)
    return out


def build_positivity(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Boolean cells x gated-columns matrix, plus the measurement names."""
    gated = gated_sign_columns(df)
    if not gated:
        return np.zeros((len(df), 0), dtype=bool), []
    mat = np.column_stack(
        [
            df[c].astype("string").fillna("").str.strip().eq("+").to_numpy()
            for c in gated
        ]
    )
    return mat, [c[: -len(SIGN_SUFFIX)] for c in gated]


def read_flowpath_csv(path: str) -> pd.DataFrame:
    """Read a FlowPath phenotype export, rejecting anything that is not one.

    Parameters
    ----------
    path : str
        CSV written by FlowPath's ``PhenotypeCsvExporter``.

    Returns
    -------
    pandas.DataFrame
        The table as read.

    Raises
    ------
    ValueError
        If any of ``phenotype``, ``centroid_x``, ``centroid_y`` is absent -- the
        three columns the join depends on.
    """
    df = pd.read_csv(path)
    missing = [
        c for c in ("phenotype", "centroid_x", "centroid_y") if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{path} does not look like a FlowPath phenotype export "
            f"(missing {missing}). Expected the PhenotypeCsvExporter schema."
        )
    return df


# ── joining ────────────────────────────────────────────────────────────────────
def join_by_label(flow: pd.DataFrame, labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Exact join on the segmentation label. Returns row indices into ``flow``."""
    lut = {int(v): i for i, v in enumerate(flow[INSTANCE_KEY].to_numpy())}
    idx = np.array([lut.get(int(v), -1) for v in labels], dtype=np.int64)
    hit = int((idx >= 0).sum())
    return idx, {
        "method": "label",
        "matched": hit,
        "cells": int(len(labels)),
        "match_fraction": hit / max(len(labels), 1),
    }


def join_by_centroid(
    flow: pd.DataFrame,
    centroids_px: np.ndarray,
    pixel_size: float,
    max_dist_px: float,
) -> Tuple[np.ndarray, Dict]:
    """Mutual-nearest centroid join, used when FlowPath exported no ``label``.

    Both sides are corner-of-pixel, so the inverse is a plain division and is
    exact. FlowPath's micrometre centroids come from the ``Centroid X µm``
    measurement ``export_geojson`` wrote, which converts; ``centroids_px`` is
    the store's ``obsm['spatial']``, which ``build_table`` converts by the same
    rule (``bin/utils/pixel_convention.py``).

    Do NOT reintroduce a half-pixel correction here. One used to live in this
    function, correct only while ``obsm['spatial']`` was still raw
    centre-of-pixel regionprops output; once the store was normalised to corner
    it became a constant 0.707 px bias on every cell -- 7% of the default
    ``--centroid-join-max-px`` budget spent on nothing, narrowing the margin
    between a true match and its neighbour.

    Matching is **mutual** nearest-neighbour: a FlowPath cell and a table cell
    must each be the other's closest, which prevents one dense cluster from
    absorbing several rows.
    """
    from scipy.spatial import cKDTree

    flow_px = flow[["centroid_x", "centroid_y"]].to_numpy(dtype=float) / pixel_size

    tree_flow = cKDTree(flow_px)
    tree_cell = cKDTree(centroids_px)
    d_fwd, i_fwd = tree_flow.query(centroids_px, distance_upper_bound=max_dist_px)
    _, i_back = tree_cell.query(flow_px, distance_upper_bound=max_dist_px)

    idx = np.full(len(centroids_px), -1, dtype=np.int64)
    for cell_i, (d, flow_i) in enumerate(zip(d_fwd, i_fwd)):
        if not np.isfinite(d) or flow_i >= len(flow_px):
            continue
        if i_back[flow_i] == cell_i:  # mutual
            idx[cell_i] = flow_i

    hit = int((idx >= 0).sum())
    return idx, {
        "method": "centroid",
        "max_dist_px": float(max_dist_px),
        "pixel_size_um": float(pixel_size),
        "matched": hit,
        "cells": int(len(centroids_px)),
        "match_fraction": hit / max(len(centroids_px), 1),
    }


def resolve_join(
    flow: pd.DataFrame,
    labels: np.ndarray,
    centroids_px: Optional[np.ndarray],
    pixel_size: float,
    max_dist_px: float,
    min_match_fraction: float,
) -> Tuple[np.ndarray, Dict]:
    """Pick the safest available join and refuse a bad one."""
    if INSTANCE_KEY in flow.columns:
        idx, stats = join_by_label(flow, labels)
        logger.info("  join: label, %d/%d matched", stats["matched"], stats["cells"])
    elif centroids_px is not None and len(centroids_px):
        logger.warning(
            "  FlowPath CSV has no '%s' column — falling back to a centroid join. "
            "Add the label measurement to the GeoJSON export to make this exact.",
            INSTANCE_KEY,
        )
        idx, stats = join_by_centroid(flow, centroids_px, pixel_size, max_dist_px)
        logger.info("  join: centroid, %d/%d matched", stats["matched"], stats["cells"])
    else:
        raise ValueError(
            "cannot join: the FlowPath CSV has no 'label' column and the table has no "
            "spatial coordinates. Refusing to align positionally — FlowPath's cell_id is "
            "a collection index, not a cell identity."
        )

    if stats["match_fraction"] < min_match_fraction:
        raise ValueError(
            f"only {stats['match_fraction']:.1%} of cells matched (threshold "
            f"{min_match_fraction:.0%}). This usually means the CSV belongs to a "
            f"different patient, or --pixel-size is wrong for a centroid join. "
            f"Refusing to write a mostly-unlabelled dataset; override with "
            f"--min-match-fraction if this is expected."
        )
    return idx, stats


def apply_flowpath(adata, flow: pd.DataFrame, idx: np.ndarray, stats: Dict) -> None:
    """Write phenotype, flags, positivity and provenance onto an AnnData in place."""
    ok = idx >= 0
    take = np.where(ok, idx, 0)

    pheno = flow["phenotype"].astype("string").fillna("").to_numpy()[take]
    pheno = np.where(ok, pheno, "")
    pheno = np.where(np.char.strip(pheno.astype(str)) == "", "unclassified", pheno)
    adata.obs["phenotype"] = pd.Categorical(pheno)

    for src, dst in (
        ("Outlier", "fp_outlier"),
        ("Out_of_annotation", "fp_out_of_annotation"),
    ):
        if src in flow.columns:
            vals = flow[src].fillna(False).astype(bool).to_numpy()[take]
            adata.obs[dst] = np.where(ok, vals, False)

    adata.obs["fp_matched"] = ok

    mat, names = build_positivity(flow)
    if names:
        aligned = np.zeros((len(idx), mat.shape[1]), dtype=bool)
        aligned[ok] = mat[idx[ok]]
        adata.obsm["positivity"] = aligned
        adata.uns["positivity_columns"] = names
        logger.info("  positivity: %d gated columns", len(names))
    else:
        logger.warning("  no gated columns in the FlowPath export; positivity omitted")

    adata.uns.setdefault("flowpath", {})["join"] = stats


# ── per-patient processing ─────────────────────────────────────────────────────
def load_table(zarr_path: str):
    """Open a MIRAGE SpatialData store and return it with its cell table.

    Parameters
    ----------
    zarr_path : str
        Path to a ``.zarr`` written by ``export_spatialdata.py``.

    Returns
    -------
    tuple
        ``(sdata, table)`` -- the ``SpatialData`` object and its ``tables['table']``.

    Raises
    ------
    ValueError
        If the store has no ``table`` element.
    """
    import spatialdata as sd

    sdata = sd.read_zarr(zarr_path)
    if "table" not in sdata.tables:
        raise ValueError(f"{zarr_path} has no 'table' element")
    return sdata, sdata.tables["table"]


def _process_one(zarr_path: str, csv_path: str, args) -> Tuple[object, Dict]:
    """Join one patient and return its (updated) AnnData plus join stats."""
    logger.info("Patient store: %s", zarr_path)
    sdata, table = load_table(zarr_path)

    if INSTANCE_KEY not in table.obs:
        raise ValueError(f"{zarr_path} table has no '{INSTANCE_KEY}' obs column")
    labels = table.obs[INSTANCE_KEY].to_numpy()
    centroids = table.obsm.get("spatial")

    flow = read_flowpath_csv(csv_path)
    logger.info("  FlowPath CSV: %s (%d rows)", csv_path, len(flow))

    idx, stats = resolve_join(
        flow,
        labels,
        np.asarray(centroids, dtype=float) if centroids is not None else None,
        args.pixel_size,
        args.centroid_join_max_px,
        args.min_match_fraction,
    )
    stats["zarr"] = str(zarr_path)
    stats["flowpath_csv"] = str(csv_path)

    apply_flowpath(table, flow, idx, stats)

    if args.gate_tree:
        table.uns.setdefault("flowpath", {})["gate_tree"] = Path(
            args.gate_tree
        ).read_text()

    if args.update_zarr:
        sdata.delete_element_from_disk("table")
        sdata.write_element("table")
        logger.info("  updated %s in place", zarr_path)

    return table, stats


def pair_inputs(zarrs: List[str], csvs: List[str]) -> List[Tuple[str, str]]:
    """Pair stores with CSVs by patient stem, falling back to positional order.

    Stem pairing is preferred because a mis-paired patient is the one failure mode
    the label join cannot catch on its own (it would simply match nothing, and the
    min-match-fraction guard would fire — but a clear pairing error is friendlier).
    """
    if len(zarrs) == 1 and len(csvs) == 1:
        return [(zarrs[0], csvs[0])]

    by_stem = {Path(z).stem: z for z in zarrs}
    pairs, unmatched = [], []
    for c in csvs:
        stem = Path(c).stem
        hit = next(
            (s for s in by_stem if s and (s in stem or stem.startswith(s))), None
        )
        if hit:
            pairs.append((by_stem[hit], c))
        else:
            unmatched.append(c)

    if unmatched:
        if len(zarrs) != len(csvs):
            raise SystemExit(
                f"cannot pair inputs: {len(zarrs)} stores and {len(csvs)} CSVs, and "
                f"these CSVs match no store by name: {unmatched}"
            )
        logger.warning("pairing by position — no name match for %s", unmatched)
        return list(zip(sorted(zarrs), sorted(csvs)))
    return pairs


# ── cohort assembly ────────────────────────────────────────────────────────────
def _concat_cohort(tables: List[object]):
    import anndata as ad

    if len(tables) == 1:
        return tables[0]
    # `join="outer"` because panels can differ between patients in a cohort; a cell
    # that was never stained for a marker gets NaN, not a silently dropped column.
    return ad.concat(
        tables,
        join="outer",
        label="sample",
        keys=[f"s{i}" for i in range(len(tables))],
        index_unique="-",
        merge="unique",
        uns_merge="unique",
    )


def flatten(adata) -> pd.DataFrame:
    """One row per cell: obs + spatial + intensities + z-scores + positivity."""
    df = adata.obs.copy().reset_index(drop=True)
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=float)
        df["x_px"], df["y_px"] = xy[:, 0], xy[:, 1]

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    for j, name in enumerate(adata.var_names):
        df[str(name)] = X[:, j]

    if "zscore" in adata.layers:
        Z = np.asarray(adata.layers["zscore"])
        for j, name in enumerate(adata.var_names):
            df[f"{name}{ZSCORE_SUFFIX}"] = Z[:, j]

    if "positivity" in adata.obsm:
        P = np.asarray(adata.obsm["positivity"], dtype=bool)
        for j, name in enumerate(adata.uns.get("positivity_columns", [])):
            df[f"{name}_positive"] = P[:, j]

    if "qc_reg_residual_px" in adata.obsm:
        R = np.asarray(adata.obsm["qc_reg_residual_px"], dtype=float)
        for j, name in enumerate(adata.uns.get("qc_reg_residual_columns", [])):
            df[f"qc_reg_residual_px__{name}"] = R[:, j]
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """Parse the FlowPath-join CLI.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        ``zarr`` and ``flowpath`` (parallel lists, paired by ``pair_inputs``),
        plus the output selectors ``out_h5ad`` / ``out_table`` / ``out_stats`` /
        ``update_zarr``.
    """
    ap = argparse.ArgumentParser(
        description="Join MIRAGE SpatialData stores with FlowPath phenotype CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # one patient, write the joined cohort table\n"
            "  join_flowpath.py --zarr results/P001/spatialdata/P001.zarr \\\n"
            "                   --flowpath P001_phenotypes.csv \\\n"
            "                   --out-h5ad cohort.h5ad --out-table cohort.parquet\n\n"
            "  # a whole cohort, also writing phenotypes back into each store\n"
            "  join_flowpath.py --zarr results/*/spatialdata/*.zarr \\\n"
            "                   --flowpath phenotypes/*.csv \\\n"
            "                   --out-h5ad cohort.h5ad --update-zarr\n"
        ),
    )
    ap.add_argument("--zarr", nargs="+", required=True, help="SpatialData store(s)")
    ap.add_argument(
        "--flowpath", nargs="+", required=True, help="FlowPath phenotype CSV(s)"
    )
    ap.add_argument("--gate-tree", help="FlowPath gate-tree JSON, stored as provenance")
    ap.add_argument("--out-h5ad", help="write the concatenated cohort AnnData here")
    ap.add_argument(
        "--out-table", help="write a flat per-cell table (.csv or .parquet)"
    )
    ap.add_argument("--out-stats", help="write join statistics as JSON")
    ap.add_argument(
        "--update-zarr",
        action="store_true",
        help="also write phenotype/positivity back into each source store",
    )
    ap.add_argument(
        "--pixel-size", type=str, default=None, help="µm per pixel, or 'auto'"
    )
    ap.add_argument(
        "--centroid-join-max-px",
        type=float,
        default=10.0,
        help="max centroid distance when joining without a label column",
    )
    ap.add_argument(
        "--min-match-fraction",
        type=float,
        default=0.5,
        help="fail if fewer than this fraction of cells match",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point: join each patient's store with its FlowPath CSV and write the cohort.

    Returns
    -------
    int
        0 on success.

    Raises
    ------
    SystemExit
        If none of ``--out-h5ad``, ``--out-table``, ``--update-zarr`` was given --
        the run would do work and write nothing.
    """
    configure_logging(level=logging.INFO)
    a = parse_args(argv)

    if not (a.out_h5ad or a.out_table or a.update_zarr):
        raise SystemExit(
            "nothing to do: pass at least one of --out-h5ad, --out-table, --update-zarr"
        )

    pairs = pair_inputs(list(a.zarr), list(a.flowpath))
    logger.info("Joining %d patient(s)", len(pairs))

    tables, all_stats = [], []
    for zarr_path, csv_path in pairs:
        table, stats = _process_one(zarr_path, csv_path, a)
        tables.append(table)
        all_stats.append(stats)

    cohort = _concat_cohort(tables)
    logger.info("Cohort: %d cells x %d measurements", cohort.n_obs, cohort.n_vars)

    if a.out_h5ad:
        Path(a.out_h5ad).parent.mkdir(parents=True, exist_ok=True)
        cohort.write_h5ad(a.out_h5ad)
        logger.info("Wrote %s", a.out_h5ad)

    if a.out_table:
        df = flatten(cohort)
        out = Path(a.out_table)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        logger.info("Wrote %s (%d rows x %d columns)", out, len(df), df.shape[1])

    if a.out_stats:
        Path(a.out_stats).write_text(json.dumps(all_stats, indent=2, default=str))
        logger.info("Wrote %s", a.out_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
