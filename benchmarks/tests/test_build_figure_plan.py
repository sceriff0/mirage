"""build_figure_plan: configs/figures.yaml -> the rows submit_figures.sh runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from benchmarks import build_figure_plan as bfp

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "benchmarks" / "configs" / "figures.yaml"


def _cfg(**over):
    base = {
        "arms": ["valis_high_micro2"],
        "reference_arm": "valis_high_micro2",
        "segmentation": {"methods": ["stardist"]},
        "figures": {},
    }
    base.update(over)
    return base


def _kinds(rows):
    out: dict[str, int] = {}
    for row in rows:
        out[row[0]] = out.get(row[0], 0) + 1
    return out


# --- the size axes are crossed, the expensive ones are not -------------------------
def test_overlay_crosses_arm_field_and_zoom():
    rows = bfp.plan(
        _cfg(
            arms=["valis_high_micro2", "stare_high"],
            figures={
                "overlay": {"field_um": [500, 2000], "zoom_um": [0, 60], "variants": 3}
            },
        )
    )
    over = [r for r in rows if r[0] == "overlay"]
    assert len(over) == 2 * 2 * 2
    assert all(r[4] == 3 for r in over)  # variants ride along, they are not an axis
    assert {(r[2], r[3]) for r in over} == {
        (500.0, 0.0),
        (500.0, 60.0),
        (2000.0, 0.0),
        (2000.0, 60.0),
    }


def test_zoom_and_crop_cross_method_field_and_mask():
    rows = bfp.plan(
        _cfg(
            segmentation={"methods": ["stardist", "instantseg"]},
            figures={
                "zoom": {
                    "field_um": [150, 300],
                    "masks": ["both", "cell"],
                    "crop": "also",
                },
                "crop": {"crop_px": [1024, 2048]},
            },
        )
    )
    k = _kinds(rows)
    assert k["zoom"] == 2 * 2 * 2  # method x field x mask
    assert k["crop"] == k["zoom"] * 2  # x output size


def test_channel_crops_are_drawn_once_off_the_reference_arm():
    """A channel crop needs no segmentation and no second arm: crossing it over either would
    draw the same picture again."""
    rows = bfp.plan(
        _cfg(
            arms=["valis_high_micro2", "stare_high"],
            segmentation={"methods": ["stardist", "instantseg"]},
            figures={
                "channels": {
                    "names": ["DAPI", "CD3"],
                    "colors": ["white", "#00e5ff"],
                    "field_um": [150],
                    "crop_px": [1024],
                }
            },
        )
    )
    chan = [r for r in rows if r[0] == "channel"]
    assert len(chan) == 2
    assert {r[1] for r in chan} == {"valis_high_micro2"}
    assert [r[5] for r in chan] == ["white", "#00e5ff"]


def test_one_mosaic_row_per_patch_size_covering_every_arm():
    rows = bfp.plan(
        _cfg(
            arms=["valis_high_micro2", "stare_high", "ashlar"],
            figures={"mosaic": {"patch_um": [200, 400], "variants": 2}},
        )
    )
    mosaic = [r for r in rows if r[0] == "mosaic"]
    assert [(r[1], r[2]) for r in mosaic] == [(200.0, 2), (400.0, 2)]


def test_rows_are_ordered_by_what_they_depend_on():
    """The launcher registers, then segments, then draws: a row must never come before the
    run it reads."""
    rows = bfp.plan(
        _cfg(
            segmentation={"methods": ["stardist", "instantseg"]},
            figures={
                "mosaic": {"patch_um": [200]},
                "overlay": {"field_um": [500], "zoom_um": [0]},
                "zoom": {"field_um": [150], "masks": ["cell"], "crop": "none"},
                "channels": {"names": ["DAPI"], "field_um": [150], "crop_px": [512]},
            },
        )
    )
    order = [r[0] for r in rows]
    assert order.index("mosaic") < order.index("overlay") < order.index("zoom")
    assert order.index("zoom") < order.index("channel")
    seg_rows = [r[1] for r in rows if r[0] == "zoom"]
    assert seg_rows == sorted(seg_rows, key=lambda m: seg_rows.index(m))  # method-major


# --- validation, at plan time ------------------------------------------------------
def test_an_unknown_arm_or_method_fails_here_not_in_the_job():
    with pytest.raises(SystemExit, match="arms:.*unknown"):
        bfp.plan(_cfg(arms=["valis_low"]))
    with pytest.raises(SystemExit, match="not accepted by the pipeline"):
        bfp.plan(_cfg(segmentation={"methods": ["cellpose"]}))


def test_the_allowed_methods_come_from_the_schema_not_a_copy():
    schema = json.loads((REPO / "nextflow_schema.json").read_text())

    def find(node):
        if isinstance(node, dict):
            if isinstance(node.get("seg_method"), dict):
                return tuple(node["seg_method"]["enum"])
            for v in node.values():
                got = find(v)
                if got:
                    return got
        return None

    assert bfp._schema_seg_methods() == find(schema)


def test_a_reference_arm_outside_the_arm_list_is_refused():
    with pytest.raises(SystemExit, match="reference_arm"):
        bfp.plan(_cfg(arms=["stare_high"], reference_arm="valis_high_micro2"))


def test_zoom_without_a_segmentation_method_is_refused():
    """It draws segmented cells: without a method there is nothing to draw them from."""
    with pytest.raises(SystemExit, match="needs segmentation.methods"):
        bfp.plan(
            _cfg(segmentation={"methods": []}, figures={"zoom": {"field_um": [150]}})
        )


def test_nonsense_axes_are_refused():
    for figures, match in (
        ({"overlay": {"field_um": []}}, "non-empty"),
        ({"overlay": {"field_um": ["wide"]}}, "not a number"),
        ({"overlay": {"field_um": [0]}}, "must be positive"),
        ({"overlay": {"field_um": [500], "variants": 0}}, "must be >= 1"),
        ({"zoom": {"field_um": [150], "masks": ["membrane"]}}, "masks"),
        ({"zoom": {"field_um": [150], "crop": "maybe"}}, "crop"),
        (
            {"channels": {"names": [""], "field_um": [150], "crop_px": [512]}},
            "expected names",
        ),
    ):
        with pytest.raises(SystemExit, match=match):
            bfp.plan(_cfg(figures=figures))


def test_a_zoom_um_of_zero_is_allowed_it_means_no_inset():
    rows = bfp.plan(_cfg(figures={"overlay": {"field_um": [500], "zoom_um": [0]}}))
    assert [r[3] for r in rows] == [0.0]


# --- the shipped config ------------------------------------------------------------
def test_the_shipped_config_expands_to_every_kind():
    rows = bfp.plan(bfp.load(SHIPPED))
    assert set(_kinds(rows)) == {"mosaic", "overlay", "zoom", "crop", "channel"}


def test_the_shipped_plan_is_tsv_the_shell_can_read():
    text = bfp.format_rows(bfp.plan(bfp.load(SHIPPED)))
    for line in text.splitlines():
        fields = line.split("\t")
        assert fields[0] in ("mosaic", "overlay", "zoom", "crop", "channel")
        assert all(f != "" for f in fields)
    assert "\t150\t" in text and ".0\t" not in text  # plain numbers, not 150.0


def test_the_shipped_config_names_only_settings_the_launcher_reads():
    """An option nobody reads is a setting that silently does nothing."""
    cfg = yaml.safe_load(SHIPPED.read_text())
    launcher = (REPO / "benchmarks" / "submit_figures.sh").read_text()
    for key in cfg.get("options") or {}:
        assert f"read_opt options.{key} " in launcher, key


def test_the_summary_names_the_expensive_axes():
    cfg = bfp.load(SHIPPED)
    text = bfp.summary(cfg, bfp.plan(cfg))
    assert "registration arm(s)" in text and "segmentation run(s)" in text
