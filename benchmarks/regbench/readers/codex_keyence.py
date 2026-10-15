"""Keyence CODEX cycle (a ``CycN_regR.zip`` or the folder it holds) -> one tile-series cycle.

Layout (S-BIAD859, checked 2026-10-14): members ``Cyc11_reg4/4_00051_Z003_CH2.tif`` =
``<reg>_<tile5>_Z<zzz>_CH<n>.tif``; ``Experiment.json`` has ``region_width``, ``region_height``,
``tile_width``, ``tile_height``, ``tile_overlap_X``/``_Y`` (PIXELS), ``tiling_mode "snake"``,
``num_z_planes``, ``per_pixel_XY_resolution`` (nm). Channel names are in ``channelnames.txt``
beside ``Experiment.json`` (the sample folder), HOECHST first within a cycle.

The zips are stored uncompressed: each member is streamed with ``ZipFile.open`` (one plane
in memory at a time), never extracted. ``.bzcfg`` is not read.

Not verified on real files: how many names ``channelnames.txt`` holds. A file with a
multiple of the cycle's channel count (more than one cycle) is sliced by the cycle number
in the zip name; a file with exactly the channel count is taken whole.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

import tifffile

from .. import tiles
from ._common import BestFocus, snake_rowcol

TILE_RE = re.compile(r"^(\d+)_(\d+)_Z(\d+)_CH(\d+)\.tif$", re.IGNORECASE)
CYCLE_RE = re.compile(r"Cyc(\d+)_reg(\d+)", re.IGNORECASE)
NUCLEAR_CH = 1  # HOECHST is first within a cycle


def pick_channel_names(files: dict[str, str]) -> list[str] | None:
    """``files``: ``{file name: text}`` for the sample folder's ``channelnames.txt`` spellings.
    Refuses a folder with two spellings whose content differs."""
    texts = {n: t for n, t in files.items() if n.lower() == "channelnames.txt"}
    if not texts:
        return None
    lines = {n: [x.strip() for x in t.splitlines() if x.strip()] for n, t in texts.items()}
    if len({tuple(v) for v in lines.values()}) > 1:
        raise ValueError(
            f"sample folder holds {sorted(texts)} with different content: refused"
        )
    return next(iter(lines.values()))


def _sample_channel_names(folder: Path) -> list[str] | None:
    files = {}
    for n in os.listdir(folder):  # exact spellings: both may exist on a case-sensitive FS
        if n.lower() == "channelnames.txt":
            files[n] = (folder / n).read_text()
    return pick_channel_names(files)


class _Source:
    """The members of a zip or a folder, by (tile, z) for the nuclear channel."""

    def __init__(self, src: Path):
        self.src = src
        self.zf = zipfile.ZipFile(src) if src.is_file() else None
        self.members: dict[int, dict[int, object]] = {}
        self.chans = set()
        entries = (
            [(i.filename, i) for i in self.zf.infolist() if not i.is_dir()]
            if self.zf
            else [(str(p), p) for p in sorted(src.rglob("*.tif")) if p.is_file()]
        )
        for name, ref in entries:
            m = TILE_RE.match(Path(name).name)
            if not m:
                continue
            self.chans.add(int(m[4]))
            if int(m[4]) == NUCLEAR_CH:
                self.members.setdefault(int(m[2]), {})[int(m[3])] = ref

    def read(self, ref):
        if self.zf is None:
            return tifffile.imread(ref)
        # streamed: one member, never extracted. The handle goes to tifffile as it is: a
        # BytesIO copy of the member ends up in reference cycles that outlive the call
        # (measured: ~20 planes of garbage before the collector ran)
        with self.zf.open(ref) as fh:
            return tifffile.imread(fh)

    def close(self):
        if self.zf:
            self.zf.close()


def export(cycle_zip_or_dir, experiment_json, dest) -> tiles.Cycle:
    src, exp_path, dest = Path(cycle_zip_or_dir), Path(experiment_json), Path(dest)
    exp = json.loads(exp_path.read_text())
    mode = exp.get("tiling_mode")
    if mode != "snake":
        raise ValueError(f"Experiment.json tiling_mode {mode!r}: only 'snake' is read")
    n_cols, n_rows = int(exp["region_width"]), int(exp["region_height"])
    pitch_x = float(exp["tile_width"]) - float(exp["tile_overlap_X"])
    pitch_y = float(exp["tile_height"]) - float(exp["tile_overlap_Y"])
    names_all = _sample_channel_names(exp_path.parent)  # refuses a conflicting pair
    source = _Source(src)
    try:
        n_tiles = n_cols * n_rows
        absent = sorted(set(range(1, n_tiles + 1)) - set(source.members))
        if absent:
            raise ValueError(f"{src}: no nuclear-channel files for tile(s) {absent[:5]}")
        per = max(source.chans)
        m = CYCLE_RE.search(src.name)
        cyc = int(m[1]) if m else 1
        if names_all and len(names_all) == per:
            names = names_all
        elif names_all and len(names_all) % per == 0 and len(names_all) >= cyc * per:
            names = names_all[(cyc - 1) * per : cyc * per]
        else:
            names = [f"CH{c}" for c in sorted(source.chans)]
        chosen: dict[str, int] = {}

        def planes():
            for tile in range(1, n_tiles + 1):
                best = BestFocus()
                for z, ref in sorted(source.members[tile].items()):
                    best.offer(z, source.read(ref))  # one plane; only the best is kept
                chosen[str(tile)] = best.z
                row, col = snake_rowcol(tile - 1, n_cols)
                yield tile, best.plane, col * pitch_x, row * pitch_y
                best = None

        return tiles.write_cycle(
            dest,
            planes(),
            pixel_size_um=float(exp["per_pixel_XY_resolution"]) / 1000.0,
            channels=names,
            nuclear_channel=0,
            extra={
                "source": str(src),
                "reader": "codex_keyence",
                "focus": "argmax_z variance of the Laplacian",
                "z": chosen,
            },
        )
    finally:
        source.close()
