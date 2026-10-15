"""PhenoCycler (Akoya) ``MICROSCOPE_OUTPUT`` -> one tile-series cycle per ``cycNNN_regRRR``.

Layout (Zenodo 6461611, checked against its files 2026-10-14): ``cyc00N_reg001/`` holds
``<region>_<tile5>_Z<zzz>_CH<c>.tif``; ``experiment.json`` has ``regionWidth``, ``regionHeight``,
``tileWidth``, ``tileHeight``, ``tileOverlapX/Y`` (a FRACTION), ``tilingMode "snakerows"``,
``numZPlanes``, ``xyResolution`` (nm) and ``channelNames.channelNamesArray`` (every cycle's
names, DAPI first within a cycle).

Tile ``n`` (1-based) sits at snake (row, col); x = col * tileWidth * (1 - tileOverlapX),
y = row * tileHeight * (1 - tileOverlapY). Per tile the nuclear channel's best-focus z is
the argmax over z of the variance of the Laplacian; the chosen z of every tile goes into
``cycle.json`` (``z``).

Where ``experiment.json`` sits relative to ``MICROSCOPE_OUTPUT`` was not checked: the root
itself, ``MICROSCOPE_OUTPUT`` and the root's parent are searched in that order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import tifffile

from .. import tiles
from ._common import BestFocus, snake_rowcol

TILE_RE = re.compile(r"^(\d+)_(\d+)_Z(\d+)_CH(\d+)\.tif$", re.IGNORECASE)
CYCLE_RE = re.compile(r"^cyc(\d+)_reg(\d+)$", re.IGNORECASE)
NUCLEAR_CH = 1  # DAPI is first within a cycle


def _find_experiment(root: Path) -> Path:
    for d in (root, root / "MICROSCOPE_OUTPUT", root.parent):
        if (d / "experiment.json").is_file():
            return d / "experiment.json"
    raise FileNotFoundError(f"no experiment.json under {root}")


def _grid(exp: dict):
    mode = exp.get("tilingMode")
    if mode != "snakerows":
        raise ValueError(f"experiment.json tilingMode {mode!r}: only 'snakerows' is read")
    w, h = int(exp["regionWidth"]), int(exp["regionHeight"])
    pitch_x = float(exp["tileWidth"]) * (1.0 - float(exp["tileOverlapX"]))
    pitch_y = float(exp["tileHeight"]) * (1.0 - float(exp["tileOverlapY"]))
    return w, h, pitch_x, pitch_y


def _export_cycle(cdir: Path, exp: dict, names, dest: Path) -> tiles.Cycle:
    n_cols, n_rows, pitch_x, pitch_y = _grid(exp)
    by_tile: dict[int, dict[int, Path]] = {}
    chans = set()
    for p in sorted(cdir.iterdir()):
        m = TILE_RE.match(p.name)
        if not m:
            continue
        chans.add(int(m[4]))
        if int(m[4]) == NUCLEAR_CH:
            by_tile.setdefault(int(m[2]), {})[int(m[3])] = p
    n_tiles = n_cols * n_rows
    absent = sorted(set(range(1, n_tiles + 1)) - set(by_tile))
    if absent:
        raise ValueError(f"{cdir}: no nuclear-channel files for tile(s) {absent[:5]}")
    chosen: dict[str, int] = {}

    def planes():
        for tile in range(1, n_tiles + 1):
            best = BestFocus()
            for z, path in sorted(by_tile[tile].items()):
                best.offer(z, tifffile.imread(path))  # one plane; only the best is kept
            chosen[str(tile)] = best.z
            row, col = snake_rowcol(tile - 1, n_cols)
            yield tile, best.plane, col * pitch_x, row * pitch_y
            best = None

    return tiles.write_cycle(
        dest,
        planes(),
        pixel_size_um=float(exp["xyResolution"]) / 1000.0,
        channels=names,
        nuclear_channel=0,
        extra={
            "source": str(cdir),
            "reader": "phenocycler",
            "focus": "argmax_z variance of the Laplacian",
            "z": chosen,
        },
    )


def export(root, dest) -> list[tiles.Cycle]:
    """Every ``cycNNN_regRRR`` folder under ``root`` -> ``dest/<folder>/``, as cycles."""
    root, dest = Path(root), Path(dest)
    exp_path = _find_experiment(root)
    exp = json.loads(exp_path.read_text())
    base = root / "MICROSCOPE_OUTPUT" if (root / "MICROSCOPE_OUTPUT").is_dir() else root
    cdirs = sorted(p for p in base.iterdir() if p.is_dir() and CYCLE_RE.match(p.name))
    if not cdirs:
        raise FileNotFoundError(f"{base}: no cycNNN_regRRR folders")
    all_names = list((exp.get("channelNames") or {}).get("channelNamesArray") or [])
    out = []
    for cdir in cdirs:
        chans = {
            int(m[4])
            for p in cdir.iterdir()
            if (m := TILE_RE.match(p.name))
        }
        per = max(chans)
        cyc = int(CYCLE_RE.match(cdir.name)[1])
        names = all_names[(cyc - 1) * per : cyc * per] or [f"CH{c}" for c in sorted(chans)]
        out.append(_export_cycle(cdir, exp, names, dest / cdir.name))
    return out
