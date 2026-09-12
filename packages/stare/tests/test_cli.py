"""The ``stare`` command: subcommands dispatch, argv passes through, the row form of
``reg-tile`` addresses the tile the explicit form does.

Deliberately small. The learned COARSE anchor needs torch + kornia, and the real
coverage of the method -- the four-stage chain, the fan-out vs ``stare register``
parity, the solver's byte-for-byte legacy claim -- lives in the mirage suite
(``tests/test_tiled_fanout.py``, ``tests/test_stare_package_parity.py``) and in
``test_solve.py`` beside this file. What is pinned here needs no model: the CLI's
wiring, and the ``--plan/--row`` contract on a hand-written plan.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import tifffile
from stare import cli
from stare.stages import reg_tile


def test_stage_subcommands_pass_argv_through_to_the_stage(capsys):
    """``stare solve --help`` is the SOLVE stage's help, not the top-level parser's."""
    with pytest.raises(SystemExit) as e:
        cli.main(["solve", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--solver" in out and "--controls" in out


def test_register_rejects_a_stray_argument(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(
            [
                "register",
                "--reference",
                "r",
                "--moving",
                "m",
                "--out",
                "o",
                "--manifest",
                "j",
                "--bogus",
            ]
        )
    assert e.value.code != 0
    assert "unrecognized arguments: --bogus" in capsys.readouterr().err


def test_version_is_the_package_version(capsys):
    from stare import __version__

    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def _identity_pair(tmp_path, n=96):
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 1, size=(n, n)) * 60000).astype(np.uint16)
    stack = np.stack([img, img])
    ref, mov = tmp_path / "ref.ome.tiff", tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(ref), stack, photometric="minisblack")
    tifffile.imwrite(str(mov), stack, photometric="minisblack")
    m0 = tmp_path / "m0.json"
    m0.write_text(
        json.dumps(
            {
                "M0": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "ref_h": n,
                "ref_w": n,
                "ref_name": "ref",
            }
        )
    )
    plan = tmp_path / "tiles.csv"
    with open(plan, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["ix", "iy", "cx", "cy", "x0", "y0", "x1", "y1", "rx0", "ry0", "rx1", "ry1"]
        )
        w.writerow([0, 0, 24.0, 24.0, 0, 0, 48, 48, 0, 0, 56, 56])
        w.writerow([1, 0, 72.0, 24.0, 48, 0, 96, 48, 40, 0, 96, 56])
    return ref, mov, m0, plan


def test_plan_row_and_explicit_geometry_write_the_same_control_json(tmp_path):
    ref, mov, m0, plan = _identity_pair(tmp_path)
    common = ["--reference", str(ref), "--moving", str(mov), "--m0", str(m0)]
    by_row = tmp_path / "row.json"
    assert (
        cli.main(
            [
                "reg-tile",
                *common,
                "--plan",
                str(plan),
                "--row",
                "1",
                "--out",
                str(by_row),
            ]
        )
        == 0
    )
    explicit = tmp_path / "explicit.json"
    r = reg_tile.plan_row(plan, 1)
    argv = list(common)
    for k in ("ix", "iy", "cx", "cy", "rx0", "ry0", "rx1", "ry1"):
        argv += [f"--{k}", str(r[k])]
    assert cli.main(["reg-tile", *argv, "--out", str(explicit)]) == 0
    a, b = json.loads(by_row.read_text()), json.loads(explicit.read_text())
    assert a == b
    assert (a["ix"], a["iy"]) == (1, 0)
    assert abs(a["dx"]) < 0.5 and abs(a["dy"]) < 0.5  # an identical pair: no residual


def test_plan_row_out_of_range_is_an_error(tmp_path):
    _ref, _mov, _m0, plan = _identity_pair(tmp_path)
    with pytest.raises(IndexError):
        reg_tile.plan_row(plan, 2)


def test_mixing_the_two_forms_is_refused(tmp_path, capsys):
    ref, mov, m0, plan = _identity_pair(tmp_path)
    with pytest.raises(SystemExit) as e:
        reg_tile.main(
            [
                "--reference",
                str(ref),
                "--moving",
                str(mov),
                "--m0",
                str(m0),
                "--plan",
                str(plan),
                "--row",
                "0",
                "--ix",
                "0",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
    assert e.value.code != 0
    assert "alternatives" in capsys.readouterr().err
