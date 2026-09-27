#!/usr/bin/env python3
"""Serialize MIRAGE postprocessing output into a SpatialData ``.zarr`` store.

Nothing here is computed that MIRAGE does not already produce — this is a
serializer. Four of SpatialData's five element types map onto existing artifacts:

===========  ==========================================================
element      MIRAGE artifact
===========  ==========================================================
Image        ``pyramid.ome.tiff``            (optional, see --include-image)
Labels       ``*_cell_mask.tif`` / ``*_nuclei_mask.tif``
Shapes       ``contours.json`` (+ nucleus contours)
Table        merged quantification CSV (intensities + morphology)
Points       n/a — protein imaging, no transcripts
===========  ==========================================================

Two entry points, one script:

* **pipeline mode** (default) builds the store from postprocessing artifacts.
* **attach mode** (``--attach-phenotypes``) reopens an existing store and adds
  FlowPath gating results. Phenotyping happens interactively in QuPath *after*
  the pipeline finishes, so it can never be a DAG dependency.

The table's ``instance_key`` is the segmentation ``label``. That column is the
only stable cell identifier in the pipeline, so every join here is keyed on it
and mismatches are fatal rather than silently positional.
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
from measurements import (  # noqa: E402
    COMPARTMENTS,
    MORPHOLOGY_COLS,
    STATISTICS,
    identify_marker_columns,
    is_qc_column,
)
from pixel_convention import centre_to_corner  # noqa: E402
from pixel_size import resolve_pixel_size  # noqa: E402
from reg_residuals import join_reg_residuals  # noqa: E402

logger = get_logger(__name__)

REGION_LABELS = "cell_mask"
INSTANCE_KEY = "label"
REGION_KEY = "region"


# ── measurement keys ───────────────────────────────────────────────────────────
def parse_measurement_key(key: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Split ``"CD3: Nucleus: Median"`` into ``("CD3", "Nucleus", "Median")``.

    A bare marker name (legacy / whole-cell-mean export) returns
    ``(key, None, None)`` rather than guessing, so ``var`` never asserts a
    compartment the pipeline did not actually measure. A ``"QC: ..."`` key
    (see ``bin/utils/measurements.py``) is never a marker and is excluded from
    ``markers``/``var`` entirely (``build_table`` puts it in ``obs`` instead),
    so this also short-circuits before the compartment/statistic match.

    Matching is on the *last two* tokens, because marker names may themselves
    contain ": " (e.g. an antibody clone).
    """
    if is_qc_column(key):
        return key, None, None
    for comp in COMPARTMENTS:
        for stat in STATISTICS:
            suffix = f": {comp}: {stat}"
            if key.endswith(suffix):
                marker = key[: -len(suffix)].strip()
                if marker:
                    return marker, comp, stat
    return key, None, None


# ── QC sanitizing ──────────────────────────────────────────────────────────────
def sanitize_for_uns(obj):
    """Make a QC document safe to write through AnnData → Zarr.

    Python ``None`` does not round-trip cleanly; ``bin/warp_seg_qc.py`` emits
    ``"radius_factor": None`` today. Numeric ``None`` becomes NaN, everything
    else is dropped. Also coerces numpy scalars, which AnnData writes as
    0-d arrays that read back awkwardly.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            s = sanitize_for_uns(v)
            if s is not None:
                out[str(k)] = s
        return out
    if isinstance(obj, (list, tuple)):
        cleaned = [sanitize_for_uns(v) for v in obj]
        return [v for v in cleaned if v is not None]
    if obj is None:
        return float("nan")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_qc(reg_qc_paths: List[str], versions_paths: List[str]) -> Tuple[Dict, Dict]:
    """Collect QC into a flattened dict (``uns['qc']``) and verbatim JSON strings.

    Storing both is deliberate: the flattened form is what people actually index
    into, and the verbatim string guarantees nothing is lost to flattening.
    """
    qc: Dict = {"registration": {}}
    raw: Dict = {"registration": {}}

    for p in reg_qc_paths:
        try:
            doc = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable registration QC %s: %s", p, exc)
            continue
        key = str(doc.get("moving") or Path(p).stem)
        qc["registration"][key] = sanitize_for_uns(doc)
        raw["registration"][key] = json.dumps(doc)

    versions: Dict[str, str] = {}
    for p in versions_paths:
        try:
            versions[Path(p).name] = Path(p).read_text()
        except OSError as exc:
            logger.warning("skipping unreadable versions file %s: %s", p, exc)
    return qc, {"qc_json": raw, "versions": versions}


# ── element builders ───────────────────────────────────────────────────────────
def build_shapes(contours_path: str, labels: np.ndarray, name: str):
    """Polygon GeoDataFrame from a ``{label: [[x, y], ...]}`` contours JSON.

    Indexed by ``label`` so it lines up with the table's ``instance_key``. Cells
    without a contour are omitted rather than given a placeholder geometry.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    contours = json.loads(Path(contours_path).read_text())
    geoms, idx = [], []
    for lab in labels:
        ring = contours.get(str(int(lab)))
        if not ring or len(ring) < 3:
            continue
        geoms.append(Polygon(ring))
        idx.append(int(lab))
    if not geoms:
        logger.warning("no usable contours in %s for element %r", contours_path, name)
        return None
    logger.info("  shapes/%s: %d polygons (of %d cells)", name, len(geoms), len(labels))
    return gpd.GeoDataFrame({"geometry": geoms}, index=pd.Index(idx, name=INSTANCE_KEY))


def read_mask(path: str):
    """Lazily read a label mask as a 2-D dask array, squeezing any singleton axis.

    Same read strategy as ``read_pyramid_lazy`` (dask-backed, aszarr-opened), and
    for the same reason on the *consumer* side: ``Labels2DModel.parse`` wraps the
    array structurally (dims + transformation) and never computes across labels.
    That is unlike ``quantify.py``/``mask_to_geojson.py``, which DO force a
    whole-array read because region properties and contour tracing are global — a
    cell label can straddle any tile boundary those pick. No such constraint
    applies here, so the read stays lazy rather than loading the whole mask up
    front.

    The two sites are not necessarily on the same grid, though, and that is what
    the explicit ``rechunk`` below is for. ``bin/merge_channels_pyramid.py``
    writes the pyramid tiled at ``tile_size`` (512 by default); ``bin/segment.py``
    /``bin/segment_cellsam.py`` now also write the mask tiled (``MASK_TIFF_TILE =
    1024``, see those files), so the dask array's native chunks are 1024×1024, not
    the single-row strips an untiled write used to produce. This ``rechunk`` is
    therefore a cheap 2×2-chunk merge to a common 2048 grid today, not the
    row-to-block rebuild it used to be — but it is still needed: ``spatialdata``'s
    ``sdata.write()`` passes no ``storage_options``, so ``ome_zarr.writer``
    defaults the *output* zarr's chunk grid to whatever the dask chunksize
    happens to be (``chunks_opt = level.chunksize``), and an unpinned 1024 grid
    here would otherwise propagate into the exported store instead of the 2048
    the rest of the pipeline uses. Not routed through ``tiled_io.open_lazy``: it opens this
    same underlying zarr array internally, but never returns it — only its own
    ``(c, y, x)``-tuple-indexed ``_CHW`` wrapper, paired with a ``close`` callable
    whose lifetime would then have to be threaded through this lazy dask graph
    instead of closed immediately. For a 2-D mask it would also force a leading
    ``C=1`` axis right back off. Reading the zarr store directly with
    ``da.from_zarr`` sidesteps all three and hands ``Labels2DModel.parse`` a real,
    unwrapped dask array to rechunk.
    """
    import dask.array as da
    import tifffile

    store = tifffile.imread(path, aszarr=True, level=0)
    arr = da.from_zarr(store).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D label mask in {path}, got shape {arr.shape}")
    # Re-chunk onto the pipeline's common 2048 grid (see docstring) before this array
    # ever reaches Labels2DModel.parse / sdata.write(), so the exported store's
    # chunk grid stays sane and matches the pyramid's regardless of the source
    # mask TIFF's own tile size.
    return arr.rechunk({0: 2048, 1: 2048})


def read_pyramid_lazy(path: str):
    """Lazily read the full-resolution level of a pyramidal OME-TIFF as (c, y, x).

    Dask-backed so peak memory is one chunk rather than the whole slide — the
    pyramid is the largest artifact the pipeline produces, and materializing it
    here would defeat the point of an out-of-core format.

    Not routed through ``tiled_io.open_lazy`` for the same reason ``read_mask``
    isn't: it opens the identical underlying zarr array internally but exposes
    only its own ``(c, y, x)``-tuple-indexed wrapper plus a ``close`` callable, not
    the array itself. ``Image2DModel.parse`` below builds a multiscale pyramid over
    this array via ``scale_factors``, which needs a genuine, unwrapped dask array
    (rechunk-able, numpy-slicable) — exactly what reading the zarr store directly
    with ``da.from_zarr`` gives it.
    """
    import dask.array as da
    import tifffile

    store = tifffile.imread(path, aszarr=True, level=0)
    arr = da.from_zarr(store)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected 2-D or 3-D pyramid in {path}, got {arr.shape}")
    return arr


def channel_names(path: str, n: int) -> List[str]:
    """Channel names from OME-XML, falling back to positional names."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "utils"))
        from metadata import extract_channel_names_from_ome

        names = list(extract_channel_names_from_ome(path) or [])
        if len(names) == n:
            return names
        logger.warning(
            "OME-XML lists %d channels but the array has %d; using positional names",
            len(names),
            n,
        )
    except Exception as exc:  # noqa: BLE001 - metadata is advisory, never fatal here
        logger.warning("could not read channel names from %s: %s", path, exc)
    return [f"channel_{i}" for i in range(n)]


# ── table ──────────────────────────────────────────────────────────────────────
def build_table(
    quant_csv: str,
    patient_id: str,
    reg_residuals: Optional[pd.DataFrame],
    qc: Dict,
    extras: Dict,
    residual_stats: Dict,
):
    """Assemble the AnnData table and attach the SpatialData region metadata."""
    import anndata as ad
    from spatialdata.models import TableModel

    df = pd.read_csv(quant_csv)
    if INSTANCE_KEY not in df.columns:
        raise ValueError(
            f"{quant_csv} has no '{INSTANCE_KEY}' column — it is the instance key that "
            "ties every table row to a mask label; without it the object cannot be built."
        )
    df = df.drop_duplicates(subset=INSTANCE_KEY, keep="first").reset_index(drop=True)

    markers = identify_marker_columns(df)
    if not markers:
        raise ValueError(f"no marker intensity columns found in {quant_csv}")

    X = df[markers].to_numpy(dtype=np.float32)

    parsed = [parse_measurement_key(m) for m in markers]
    var = pd.DataFrame(
        {
            "marker": [p[0] for p in parsed],
            "compartment": [p[1] for p in parsed],
            "statistic": [p[2] for p in parsed],
        },
        index=pd.Index(markers, name="measurement"),
    )

    obs = pd.DataFrame(index=pd.RangeIndex(len(df)).astype(str))
    obs[INSTANCE_KEY] = df[INSTANCE_KEY].to_numpy(dtype=np.int64)
    obs["patient_id"] = patient_id
    for col in MORPHOLOGY_COLS:
        if col in df.columns and col not in (INSTANCE_KEY, "x", "y"):
            obs[f"qc_{col}" if col not in ("fov",) else col] = df[col].to_numpy()
    # Per-cell/per-round QC columns (bin/cell_qc.py), verbatim "QC: ..." names —
    # unrelated to the `qc_<morphology>` obs columns just above, which predate this
    # spec and are morphology, not the QC vocabulary in bin/utils/measurements.py.
    for col in df.columns:
        if is_qc_column(col):
            obs[col] = df[col].to_numpy(dtype=float)
    # region_key MUST be categorical and its categories must match `region` exactly.
    obs[REGION_KEY] = pd.Categorical(
        [REGION_LABELS] * len(df), categories=[REGION_LABELS]
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)

    if {"x", "y"} <= set(df.columns):
        # Pixels, not micrometres: SpatialData's convention is intrinsic pixel
        # coordinates plus a transformation carrying the scale. Storing µm here
        # would double-apply pixel_size on any cross-modality alignment.
        #
        # Corner-of-pixel, because build_shapes' polygons already are. The
        # quantification CSV carries raw regionprops centroids, which are
        # centre-of-pixel; contours.json was converted by
        # extract_cell_properties. Storing the centroids unconverted put every
        # cell's point half a pixel up-and-left of its own polygon -- see
        # bin/utils/pixel_convention.py for the rule and who else applies it.
        adata.obsm["spatial"] = centre_to_corner(
            df[["x", "y"]].to_numpy(dtype=np.float64)
        )

    # Per-slide z-scores, matching what EXPORT_GEOJSON writes for QuPath display.
    with np.errstate(invalid="ignore", divide="ignore"):
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        adata.layers["zscore"] = np.where(
            sd > 0, (X - mu) / np.where(sd > 0, sd, 1), 0.0
        )

    if reg_residuals is not None and not reg_residuals.empty:
        aligned = reg_residuals.reindex(obs[INSTANCE_KEY].to_numpy())
        adata.obsm["qc_reg_residual_px"] = aligned.to_numpy(dtype=np.float64)
        adata.uns["qc_reg_residual_columns"] = [str(c) for c in aligned.columns]
        finite = np.isfinite(aligned.to_numpy(dtype=np.float64))
        adata.obs["qc_reg_matched"] = finite.any(axis=1)
        with np.errstate(invalid="ignore"):
            adata.obs["qc_reg_residual_max_px"] = (
                np.nanmax(
                    np.where(finite, aligned.to_numpy(dtype=np.float64), np.nan),
                    axis=1,
                    initial=np.nan,
                )
                if finite.any()
                else np.nan
            )

    adata.uns["qc"] = sanitize_for_uns(qc)
    adata.uns["qc_json"] = extras["qc_json"]
    adata.uns["qc_reg_residual_join"] = sanitize_for_uns(residual_stats)
    adata.uns["provenance"] = sanitize_for_uns(
        {"patient_id": patient_id, "versions": extras["versions"]}
    )

    return TableModel.parse(
        adata,
        region=REGION_LABELS,
        region_key=REGION_KEY,
        instance_key=INSTANCE_KEY,
    )


# ── assembly ───────────────────────────────────────────────────────────────────
def build_spatialdata(args) -> "object":
    """Assemble the SpatialData object from MIRAGE artifacts."""
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel
    from spatialdata.transformations import Identity, Scale

    px = resolve_pixel_size(
        args.pixel_size, args.pyramid, source="the merged pyramid", logger=logger
    )

    # Two coordinate systems: 'global' is intrinsic pixels (what every element is
    # stored in), 'um' scales it to micrometres. Every element carries both, so
    # they stay mutually aligned in either.
    def transforms():
        return {"global": Identity(), "um": Scale([px, px], axes=("y", "x"))}

    labels_elems, shapes_elems, images_elems = {}, {}, {}

    cell_mask = read_mask(args.cell_mask)
    labels_elems[REGION_LABELS] = Labels2DModel.parse(
        cell_mask, dims=("y", "x"), transformations=transforms()
    )
    logger.info("  labels/%s: %s", REGION_LABELS, cell_mask.shape)

    if args.nuclei_mask:
        nuc = read_mask(args.nuclei_mask)
        labels_elems["nuclei_mask"] = Labels2DModel.parse(
            nuc, dims=("y", "x"), transformations=transforms()
        )
        logger.info("  labels/nuclei_mask: %s", nuc.shape)

    quant = pd.read_csv(args.quant_csv).drop_duplicates(subset=INSTANCE_KEY)
    labels_arr = quant[INSTANCE_KEY].to_numpy(dtype=np.int64)

    cells_gdf = build_shapes(args.contours_json, labels_arr, "cells")
    if cells_gdf is not None:
        shapes_elems["cells"] = ShapesModel.parse(
            cells_gdf, transformations=transforms()
        )
    if args.nucleus_contours_json:
        nuc_gdf = build_shapes(args.nucleus_contours_json, labels_arr, "nuclei")
        if nuc_gdf is not None:
            shapes_elems["nuclei"] = ShapesModel.parse(
                nuc_gdf, transformations=transforms()
            )

    if args.include_image and args.pyramid:
        arr = read_pyramid_lazy(args.pyramid)
        names = channel_names(args.pyramid, arr.shape[0])
        images_elems["pyramid"] = Image2DModel.parse(
            arr,
            dims=("c", "y", "x"),
            c_coords=names,
            scale_factors=[2, 2, 2, 2],
            transformations=transforms(),
        )
        logger.info("  images/pyramid: %s (%d channels, lazy)", arr.shape, len(names))
    elif args.include_image:
        logger.warning("--include-image given but no --pyramid; skipping the image")

    centroids = (
        quant[["x", "y"]].to_numpy(dtype=float)
        if {"x", "y"} <= set(quant.columns)
        else np.empty((0, 2))
    )
    residuals, residual_stats = join_reg_residuals(
        args.reg_residuals or [], centroids, labels_arr, args.residual_join_max_px
    )

    qc, extras = load_qc(args.qc_json or [], args.versions or [])
    table = build_table(
        args.quant_csv, args.patient_id, residuals, qc, extras, residual_stats
    )

    return SpatialData(
        images=images_elems,
        labels=labels_elems,
        shapes=shapes_elems,
        tables={"table": table},
    )


# ── attach mode ────────────────────────────────────────────────────────────────
def _attach_phenotypes(zarr_path: str, csv_path: str, gate_tree: Optional[str]) -> None:
    """Add FlowPath gating results to an existing store.

    Joins on ``label``. FlowPath's own ``cell_id`` is a positional index into a
    QuPath detection collection whose order is not guaranteed, so a positional
    join would shift phenotype assignments and produce plausible-looking wrong
    biology rather than an error. Hence: join on ``label``, and refuse otherwise.
    """
    import spatialdata as sd

    sdata = sd.read_zarr(zarr_path)
    table = sdata.tables["table"]
    pheno = pd.read_csv(csv_path)

    if INSTANCE_KEY not in pheno.columns:
        raise ValueError(
            f"{csv_path} has no '{INSTANCE_KEY}' column. Refusing to align positionally: "
            "FlowPath's cell_id is a collection index, not a mask label, so a positional "
            "join silently mis-assigns phenotypes. Re-export with the label measurement."
        )

    pheno = pheno.drop_duplicates(subset=INSTANCE_KEY, keep="first").set_index(
        INSTANCE_KEY
    )
    order = pd.Index(table.obs[INSTANCE_KEY].to_numpy())
    missing = order.difference(pheno.index)
    if len(missing) == len(order):
        raise ValueError(
            f"no labels in {csv_path} match the table's instance key — wrong patient?"
        )
    if len(missing):
        logger.warning(
            "%d of %d cells have no phenotype row; they will be marked unclassified",
            len(missing),
            len(order),
        )

    aligned = pheno.reindex(order)
    if "phenotype" in aligned.columns:
        table.obs["phenotype"] = pd.Categorical(
            aligned["phenotype"].fillna("unclassified")
        )
    for src, dst in (
        ("Outlier", "fp_outlier"),
        ("Out_of_annotation", "fp_out_of_annotation"),
    ):
        if src in aligned.columns:
            table.obs[dst] = aligned[src].fillna(False).astype(bool).to_numpy()

    # `_sign` is three-state: "+", not-"+", or blank when no gate ever touched that
    # column. Keeping only gated columns preserves that — a boolean cast over all
    # markers would merge "negative" with "never gated".
    sign_cols = [c for c in aligned.columns if c.endswith("_sign")]
    gated = [
        c for c in sign_cols if aligned[c].notna().any() and (aligned[c] != "").any()
    ]
    if gated:
        pos = (aligned[gated] == "+").to_numpy(dtype=bool)
        table.obsm["positivity"] = pos
        table.uns["positivity_columns"] = [c[: -len("_sign")] for c in gated]
        logger.info("  positivity: %d gated columns", len(gated))

    if gate_tree:
        table.uns.setdefault("flowpath", {})["gate_tree"] = Path(gate_tree).read_text()

    sdata.delete_element_from_disk("table")
    sdata.write_element("table")
    logger.info("Attached phenotypes to %s", zarr_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """Parse the SpatialData-export CLI, which has two modes.

    Attach mode (``--attach-phenotypes`` + ``--zarr``) writes FlowPath phenotypes
    into an existing store. Pipeline mode builds a new store and requires
    ``--quant-csv``, ``--contours-json``, ``--cell-mask`` and ``--output``.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments for whichever mode was selected.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attach-phenotypes", help="FlowPath phenotype CSV (attach mode)")
    ap.add_argument("--gate-tree", help="FlowPath gate-tree JSON (attach mode)")
    ap.add_argument("--zarr", help="existing .zarr to update (attach mode)")

    ap.add_argument("--quant-csv", help="merged quantification CSV")
    ap.add_argument("--contours-json", help="whole-cell contours JSON")
    ap.add_argument("--nucleus-contours-json", help="nucleus contours JSON (re-keyed)")
    ap.add_argument("--cell-mask", help="cell label mask TIFF")
    ap.add_argument("--nuclei-mask", help="nuclei label mask TIFF")
    ap.add_argument("--pyramid", help="merged pyramidal OME-TIFF")
    ap.add_argument("--qc-json", nargs="*", help="registration QC JSONs")
    ap.add_argument("--reg-residuals", nargs="*", help="per-cell residual CSVs")
    ap.add_argument("--versions", nargs="*", help="versions.yml files")
    ap.add_argument("--patient-id", default="unknown")
    ap.add_argument("--pixel-size", type=str, default=None)
    ap.add_argument(
        "--include-image",
        action="store_true",
        help="write the pyramid into the store (duplicates the largest artifact on disk)",
    )
    ap.add_argument(
        "--residual-join-max-px",
        type=float,
        default=15.0,
        help="max centroid distance for the QC->cell_mask spatial join",
    )
    ap.add_argument("-o", "--output", help="output .zarr path")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point: attach phenotypes to an existing .zarr, or build a new one.

    Returns
    -------
    int
        0 on success. A missing mode-required argument raises ``SystemExit``.
    """
    configure_logging(level=logging.INFO)
    a = parse_args(argv)

    if a.attach_phenotypes:
        if not a.zarr:
            raise SystemExit("--attach-phenotypes requires --zarr")
        _attach_phenotypes(a.zarr, a.attach_phenotypes, a.gate_tree)
        return 0

    for req in ("quant_csv", "contours_json", "cell_mask", "output"):
        if not getattr(a, req):
            raise SystemExit(f"--{req.replace('_', '-')} is required in pipeline mode")

    logger.info("Building SpatialData for %s", a.patient_id)
    sdata = build_spatialdata(a)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sdata.write(out, overwrite=True)
    logger.info("Wrote %s", out)
    logger.info("%s", sdata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
