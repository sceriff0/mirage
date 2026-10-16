"""The outlines of a cohort slide: regbench's own StarDist cell mask, traced the way the
pipeline traces its QC outlines.

The arms' outlines were the reg_qc=2 by-product of ``valis_high_micro2``:
``SEG_QC_SEGMENT`` (``SEGMENT`` under an alias, ``subworkflows/local/seg_qc.nf``) ran
``bin/segment.py``, whose cell mask is the StarDist nuclei grown by
``seg_expand_distance`` (``bin/segment.py``: ``expand_labels_tiled``), and
``SEG_QC_GEOJSON`` traced that mask with ``bin/mask_to_geojson.py --tolerance
${simplify_tolerance}`` (``conf/modules.config``). The nuclei step
(:mod:`benchmarks.regbench.nuclei`) runs the same ``bin/segment.py`` with the same flags,
and :func:`trace` calls the same ``write_geojson`` of the same file at the same tolerance:
by construction, the outline the arm would have published from the same image, model and
settings. ``benchmarks/tests/regbench/test_outline_definition.py`` pins each link; the
equivalence is derived from the code, not measured on cohort slides.

The tolerance resolves like the StarDist settings (:mod:`site_config`): the site config's
``simplify_tolerance`` when it sets one, else ``nextflow.config``'s.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from . import site_config

REPO = Path(__file__).resolve().parents[2]
MASK_TO_GEOJSON = REPO / "bin" / "mask_to_geojson.py"
KEY = "simplify_tolerance"


def tolerance(env=None):
    """``simplify_tolerance`` as the pipeline resolves it for this site: the site
    config's value, else the repository default (``nextflow.config``)."""
    env = os.environ if env is None else env
    path = site_config.site_config_path(env)
    if path is not None:
        site = site_config.read(path, (KEY,))
        if KEY in site:
            return float(site[KEY])
    return float(site_config.read(REPO / "nextflow.config", (KEY,))[KEY])


_resolve_tolerance = tolerance  # `trace` takes a parameter of the same name


def _mask_to_geojson():
    """``bin/mask_to_geojson.py`` itself, loaded from its file. It imports ``logger``
    from ``bin/utils``: both folders are on ``sys.path`` for the import only."""
    name = "_regbench_mask_to_geojson"
    if name in sys.modules:
        return sys.modules[name]
    added = [str(MASK_TO_GEOJSON.parent), str(MASK_TO_GEOJSON.parent / "utils")]
    before = list(sys.path)
    sys.path[:0] = added
    try:
        spec = importlib.util.spec_from_file_location(name, MASK_TO_GEOJSON)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    sys.modules[name] = module
    return module


def trace(cell_mask, out_geojson, tolerance=None):
    """Trace a cell mask to the GeoJSON of its outlines with the pipeline's own
    ``write_geojson``; returns the number of outlines. The file is written next to its
    place and then moved there, so a reader never finds half of it."""
    tol = _resolve_tolerance() if tolerance is None else float(tolerance)
    out = Path(out_geojson)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    n = _mask_to_geojson().write_geojson(str(cell_mask), str(part), tol)
    os.replace(part, out)
    return int(n)
