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


def _args(row):
    import shlex

    return shlex.split(row[3])


def _flag(row, name):
    args = _args(row)
    return args[args.index(name) + 1] if name in args else None


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
    assert all(_flag(r, "--variants") == "3" for r in over)  # rides along, not an axis
    assert {(_flag(r, "--field-um"), _flag(r, "--zoom-um")) for r in over} == {
        ("500", None),
        ("500", "60"),
        ("2000", None),
        ("2000", "60"),
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


def test_the_mosaic_crosses_overlay_and_checker_and_the_numbers_source():
    rows = bfp.plan(
        _cfg(
            figures={
                "mosaic": {
                    "patch_um": [200],
                    "kinds": ["overlay", "checker"],
                    "numbers": ["image", "none"],
                }
            }
        )
    )
    mosaic = [r for r in rows if r[0] == "mosaic"]
    assert len(mosaic) == 4
    assert {(_flag(r, "--kinds"), _flag(r, "--numbers")) for r in mosaic} == {
        ("overlay", "image"),
        ("overlay", "none"),
        ("checker", "image"),
        ("checker", "none"),
    }


def test_channel_crops_cross_the_contrast_modes():
    """The comparison is drawn, not asserted in prose."""
    rows = bfp.plan(
        _cfg(
            figures={
                "channels": {
                    "names": ["DAPI"],
                    "field_um": [150],
                    "crop_px": [1024],
                    "autoscale": ["clean", "percentile"],
                }
            }
        )
    )
    chan = [r for r in rows if r[0] == "channel"]
    assert {_flag(r, "--autoscale") for r in chan} == {"clean", "percentile"}
    assert {r[2] for r in chan} == {
        "crops/channels/f150_p1024_clean",
        "crops/channels/f150_p1024_percentile",
    }


def test_several_rois_multiply_every_figure_and_land_in_their_own_directories():
    one = bfp.plan(
        _cfg(
            figures={
                "overlay": {"field_um": [500]},
                "channels": {"names": ["DAPI"], "field_um": [150], "crop_px": [512]},
            },
            options={"roi": ["100,200"]},
        )
    )
    three = bfp.plan(
        _cfg(
            figures={
                "overlay": {"field_um": [500]},
                "channels": {"names": ["DAPI"], "field_um": [150], "crop_px": [512]},
            },
            options={"roi": ["100,200", "300,400", "500,600"]},
        )
    )
    assert len(three) == 3 * len(one)
    assert all(_flag(r, "--roi") for r in three)
    assert {r[2].rsplit("_r", 1)[-1] for r in three} == {"1", "2", "3"}
    # a single ROI does not get a suffix: the directory stays the one you already have
    assert all("_r" not in r[2] for r in one)


def test_several_patients_draw_each_but_the_mosaic_takes_them_together():
    rows = bfp.plan(
        _cfg(
            figures={
                "mosaic": {"patch_um": [200]},
                "overlay": {"field_um": [500]},
            },
            options={"patient": ["033", "045"]},
        )
    )
    mosaic = [r for r in rows if r[0] == "mosaic"]
    over = [r for r in rows if r[0] == "overlay"]
    assert len(mosaic) == 1 and _args(mosaic[0]).count("--patient") == 2
    assert len(over) == 2 and {_flag(r, "--patient") for r in over} == {"033", "045"}


def test_the_drawing_options_are_decided_in_the_plan_not_the_shell():
    rows = bfp.plan(
        _cfg(
            figures={
                "zoom": {"field_um": [150], "masks": ["both"]},
                "channels": {"names": ["DAPI"], "field_um": [150], "crop_px": [512]},
            },
            options={
                "outline_color": "#ff0000",
                "outline_width": 3,
                "sat": 1.5,
                "bg_k": 4,
            },
        )
    )
    zoom = next(r for r in rows if r[0] == "zoom")
    assert _flag(zoom, "--outline-color") == "#ff0000"
    assert _flag(zoom, "--outline-width") == "3"
    chan = next(r for r in rows if r[0] == "channel")
    assert _flag(chan, "--sat") == "1.5" and _flag(chan, "--bg-k") == "4"


def test_channel_crops_are_drawn_once_off_the_reference_arm():
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
    assert {r[1] for r in chan} == {"arm:valis_high_micro2"}
    assert [_flag(r, "--colors") for r in chan] == ["white", "#00e5ff"]


def test_an_arm_that_already_exists_is_reused_not_rebuilt():
    cfg = _cfg(
        arms=[
            "valis_high_micro2",
            {"name": "arms_valis_m0", "dir": "/results/arms/valis_high_micro0"},
        ],
        figures={"overlay": {"field_um": [500]}},
    )
    rows = bfp.plan(cfg)
    assert bfp.arm_dirs(cfg) == {"arms_valis_m0": "/results/arms/valis_high_micro0"}
    assert {r[1] for r in rows if r[0] == "overlay"} == {
        "arm:valis_high_micro2",
        "arm:arms_valis_m0",
    }
    assert "1 registration arm(s) to build, 1 reused" in bfp.summary(cfg, rows)


def test_a_figure_with_no_arm_behind_it_is_titled_NA_not_blank():
    """A blank corner reads as an oversight; NA reads as a fact."""
    assert bfp.title_of("") == "NA"
    assert bfp.title_of(None) == "NA"
    assert bfp.title_of("  ") == "NA"
    assert bfp.title_of(" valis_high ") == "valis_high"


def test_rows_are_ordered_by_what_they_depend_on():
    rows = bfp.plan(
        _cfg(
            segmentation={"methods": ["stardist", "instantseg"]},
            figures={
                "mosaic": {"patch_um": [200]},
                "overlay": {"field_um": [500]},
                "zoom": {"field_um": [150], "masks": ["cell"]},
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
    assert len(rows) == 1 and _flag(rows[0], "--zoom-um") is None


# --- the shipped config ------------------------------------------------------------
def test_the_shipped_config_expands_to_every_kind():
    rows = bfp.plan(bfp.load(SHIPPED))
    assert set(_kinds(rows)) == {"mosaic", "overlay", "zoom", "crop", "channel"}


def test_the_shipped_plan_is_tsv_the_shell_can_read():
    import shlex

    text = bfp.format_rows(bfp.plan(bfp.load(SHIPPED)))
    for line in text.splitlines():
        fields = line.split("\t")
        assert len(fields) == 4, fields  # kind, run key, outdir, arguments
        kind, run, out, args = fields
        assert kind in ("mosaic", "overlay", "zoom", "crop", "channel")
        assert run.startswith(("arms:", "arm:", "seg:"))
        assert out and " " not in out  # a directory the shell can mkdir unquoted
        assert shlex.split(args)  # and arguments it can eval back
    # sizes survive as plain numbers, not 150.0 or 1.5e+02
    assert "--field-um 150 " in text


def test_every_shipped_option_is_read_by_the_plan_or_the_launcher():
    """An option nobody reads is a setting that silently does nothing. Each one is consumed
    either by the plan (it ends up in a figure's arguments) or by the shell (it drives a
    phase), and this checks that every one of them is claimed by exactly one of the two."""
    cfg = yaml.safe_load(SHIPPED.read_text())
    launcher = (REPO / "benchmarks" / "submit_figures.sh").read_text()
    planner = (REPO / "benchmarks" / "build_figure_plan.py").read_text()
    for key in cfg.get("options") or {}:
        by_shell = f"read_opt options.{key} " in launcher
        by_plan = f'"{key}"' in planner
        assert by_shell or by_plan, key


def test_the_summary_names_the_expensive_axes():
    cfg = bfp.load(SHIPPED)
    text = bfp.summary(cfg, bfp.plan(cfg))
    assert "registration arm(s)" in text and "segmentation run(s)" in text


def test_the_mosaic_row_count_is_left_to_reg_mosaic_unless_asked():
    """This file has no samplesheet, so it cannot count rounds: reg_mosaic defaults to every
    round at one ROI. Passing --rows from here without a count is what broke job 6872763."""
    rows = bfp.plan(_cfg(figures={"mosaic": {"patch_um": [200]}}))
    assert _flag(rows[0], "--rows") is None
    rows = bfp.plan(_cfg(figures={"mosaic": {"patch_um": [200], "rows": 6}}))
    assert _flag(rows[0], "--rows") == "6"


# --- mosaic groups: one comparison per figure ---------------------------------------
def test_mosaic_groups_each_draw_their_own_arms():
    """A mosaic is one column per arm, so 18 arms is a wall, not a figure. Each group is one
    axis -- the only shape in which a mosaic answers a question."""
    cfg = _cfg(
        arms=["valis_high_micro2", "stare_high", "ashlar"],
        figures={
            "mosaic": {
                "patch_um": [200],
                "groups": [
                    {"name": "backends", "arms": ["valis_high_micro2", "stare_high"]},
                    {"name": "vs_ashlar", "arms": ["valis_high_micro2", "ashlar"]},
                ],
            }
        },
    )
    rows = [r for r in bfp.plan(cfg) if r[0] == "mosaic"]
    assert len(rows) == 2
    assert [r[1] for r in rows] == [
        "arms:valis_high_micro2,stare_high",
        "arms:valis_high_micro2,ashlar",
    ]
    assert [r[2] for r in rows] == [
        "mosaic/backends_p200_overlay_auto",
        "mosaic/vs_ashlar_p200_overlay_auto",
    ]


def test_without_groups_every_arm_is_one_mosaic():
    cfg = _cfg(
        arms=["valis_high_micro2", "stare_high"],
        figures={"mosaic": {"patch_um": [200]}},
    )
    rows = [r for r in bfp.plan(cfg) if r[0] == "mosaic"]
    assert len(rows) == 1 and rows[0][1] == "arms:valis_high_micro2,stare_high"
    assert rows[0][2] == "mosaic/p200_overlay_auto"  # no group tag


def test_a_group_naming_an_arm_that_is_not_registered_is_refused():
    with pytest.raises(SystemExit, match="not in arms"):
        bfp.plan(
            _cfg(
                arms=["valis_high_micro2"],
                figures={
                    "mosaic": {
                        "patch_um": [200],
                        "groups": [{"name": "x", "arms": ["valis_low_micro0"]}],
                    }
                },
            )
        )
    with pytest.raises(SystemExit, match="needs an `arms` list"):
        bfp.plan(
            _cfg(figures={"mosaic": {"patch_um": [200], "groups": [{"name": "empty"}]}})
        )


def test_every_group_is_crossed_with_the_sizes():
    cfg = _cfg(
        arms=["valis_high_micro2", "stare_high"],
        figures={
            "mosaic": {
                "patch_um": [200, 500],
                "kinds": ["overlay", "checker"],
                "groups": [
                    {"name": "a", "arms": ["valis_high_micro2"]},
                    {"name": "b", "arms": ["stare_high"]},
                ],
            }
        },
    )
    rows = [r for r in bfp.plan(cfg) if r[0] == "mosaic"]
    assert len(rows) == 2 * 2 * 2  # groups x patches x kinds
