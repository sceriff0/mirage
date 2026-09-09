"""The VALIS tier table: one size per tier, used for both stages, and every copy agrees.

Since 2026-09-09 the `high` / `medium` / `low` rows of MEMORY_PRESETS use the SAME size
for the feature-matching working image and the non-rigid registration image:
2048 / 1024 / 512 px (maintainers' decision). Before that the non-rigid column was
4096 / 4096 / 1024, so `high` did its non-rigid stage at twice the matching size.

The numbers are restated in prose in four places an operator reads, and the AST test
here holds each of them to the table rather than to a memory of it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {"high": 2048, "medium": 1024, "low": 512}


def _presets() -> dict:
    """MEMORY_PRESETS' numeric fields, read from the source without importing valis."""
    tree = ast.parse((ROOT / "bin" / "utils" / "valis_config.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "MEMORY_PRESETS"
    )
    out = {}
    for tier_key, row in zip(node.value.keys, node.value.values):
        out[tier_key.value] = {
            k.value: v.value
            for k, v in zip(row.keys, row.values)
            if isinstance(v, ast.Constant) and isinstance(v.value, int)
        }
    return out


def test_each_tier_uses_one_size_for_both_stages():
    for tier, row in _presets().items():
        assert (
            row["max_processed_image_dim_px"]
            == row["max_non_rigid_registration_dim_px"]
            == EXPECTED[tier]
        ), (
            tier,
            row,
        )


def test_the_prose_copies_state_the_same_sizes():
    """register.py's docstring and --memory-mode help, nextflow.config's two column
    comments, and docs/parameters.md's memory_mode row all spell the table out."""
    hi, me, lo = (EXPECTED[t] for t in ("high", "medium", "low"))
    triple = f'{hi}/{hi} px, "medium" {me}/{me} px, "low" {lo}/{lo} px'
    register = (ROOT / "bin" / "register.py").read_text()
    assert f'"high" {triple}' in register, "register.py --memory-mode help is stale"
    assert (
        f'({hi}/{hi} px), "medium" ({me}/{me} px) or "low" ({lo}/{lo} px)' in register
    ), "register.py's valis_registration docstring is stale"
    config = (ROOT / "nextflow.config").read_text()
    column = f"[high: {hi}, medium: {me}, low: {lo}]"
    assert config.count(column) == 2, (
        f"nextflow.config should carry `{column}` once per reg_valis_* knob"
    )
    params = (ROOT / "docs" / "parameters.md").read_text()
    assert (
        f"`high` = {hi}/{hi} px, `medium` = {me}/{me} px, `low` = {lo}/{lo} px"
        in params
    )
    assert f"| `reg_valis_max_processed_dim` | tier (`high`: {hi})" in params
    assert f"| `reg_valis_max_non_rigid_dim` | tier (`high`: {hi})" in params


def test_no_copy_still_carries_the_old_non_rigid_column():
    """4096 was the old `high`/`medium` non-rigid size; the STARE table has its own
    4096 (a tile size), so the check is on the VALIS phrasings only."""
    stale = re.compile(r"2048/4096|1024/4096|256/1024|\[high: 4096|low: 256\]")
    for rel in (
        "bin/register.py",
        "nextflow.config",
        "docs/parameters.md",
        "docs/figures/registration-schematic.html",
        "bin/utils/valis_config.py",
    ):
        text = (ROOT / rel).read_text()
        assert not stale.search(text), (
            f"{rel} still states the pre-2026-09-09 tier sizes"
        )
