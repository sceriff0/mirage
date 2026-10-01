"""Every VALIS warp in the pipeline uses crop="reference".

REGISTER writes the registered slides in the reference frame (bin/utils/valis_config.py
crop="reference"; bin/register.py passes it explicitly), and the cell mask is segmented
on those slides. A warp in any other frame -- VALIS's "overlap" was the default of
bin/warp_seg_qc.py and bin/utils/valis_stage_warp.py until 2026-10-01 -- shifts every
point by that frame's origin: invisible to Dice and centroid displacement (both sides
shift together) but wrong for anything joined onto the cells, like the per-cell
registration residuals CELL_QC and the SpatialData export attach.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.nfmodel import strip_comments

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "bin/register.py",
    "bin/warp_seg_qc.py",
    "bin/utils/valis_stage_warp.py",
    "bin/utils/valis_config.py",
]


def _code(path: Path) -> str:
    return "\n".join(ln.split("#")[0] for ln in path.read_text().splitlines())


def test_no_valis_crop_other_than_reference():
    bad = []
    for rel in FILES:
        code = _code(ROOT / rel)
        for m in re.finditer(
            r'crop\s*=\s*([^,)\n]+)|"--crop",\s*default=([^,)\n]+)', code
        ):
            val = (m.group(1) or m.group(2)).strip()
            if val not in ('"reference"', "crop", "crop_method", "a.crop"):
                bad.append(f"{rel}: crop = {val}")
    assert not bad, 'VALIS crop must be "reference":\n' + "\n".join(bad)


def test_the_pipeline_passes_crop_reference_explicitly():
    # comments stripped, strings kept: the flag is a string literal, and a commented-out
    # one must not satisfy this
    text = strip_comments((ROOT / "lib" / "WarpBackends.groovy").read_text())
    assert '"--crop reference"' in text
