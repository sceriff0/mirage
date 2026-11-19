"""A `params/*.json` preset must be a preset that can actually run.

Three checks:

1. Every key is known to the schema. An unknown key is rejected or silently
   inert, depending on nf-schema's strictness.
2. A preset that sets a parameter owned by one segmentation backend also sets
   `seg_method` to that backend; otherwise the block is inert.
3. No preset sets `max_cpus`, `max_memory` or `max_time` (except an exempted
   test preset). `-params-file` overrides `-c`, so a ceiling in a preset would
   silently beat the site config, which owns per-site sizing.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PRESETS = sorted((REPO / "params").glob("*.json"))

# Parameter-name prefixes owned by one backend, and the seg_method value that
# activates them. Read off nextflow.config's segmentation block.
BACKEND_PREFIXES = {
    "stardist": (
        "seg_pmin",
        "seg_pmax",
        "seg_prob_thresh",
        "seg_n_tiles_x",
        "seg_n_tiles_y",
        "segmentation_model",
        "segmentation_model_dir",
    ),
    "instantseg": ("seg_instantseg_", "instanseg_model_dir"),
    "cellsam": ("seg_cellsam_", "cellsam_model_path"),
}

# conf/site.config.template owns per-site sizing; `-params-file` beats
# `-c site.config`, so a preset setting these would override the site's ceiling.
SITE_SIZING_KEYS = ("max_cpus", "max_memory", "max_time")

# Presets exempt from the site-sizing rule, with why. Each must still set at
# least one site-sizing key (checked below), so a stale entry fails.
SITE_SIZING_EXEMPT = {
    "test.json": (
        "a self-contained CI/test preset that intentionally pins tiny ceilings "
        "(2 cpus / 6.GB / 6.h) so the suite does not depend on -c site.config; "
        "not one of the four production presets R4 targets"
    ),
}


def _schema_params():
    schema = json.loads((REPO / "nextflow_schema.json").read_text())
    names = set()

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                names.update(k for k, v in props.items() if isinstance(v, dict))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert len(names) > 50, (
        f"only {len(names)} schema params found -- the walk is broken"
    )
    return names


def _schema_seg_method_enum():
    """`seg_method`'s enum in nextflow_schema.json -- the set of real backends."""
    schema = json.loads((REPO / "nextflow_schema.json").read_text())

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and "seg_method" in props:
                return props["seg_method"].get("enum")
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    enum = walk(schema)
    assert enum, "nextflow_schema.json's seg_method has no enum -- the walk is broken"
    return set(enum)


def _owner(name):
    for backend, prefixes in BACKEND_PREFIXES.items():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                return backend
    return None


def test_there_are_presets_to_check():
    assert len(PRESETS) >= 5, f"only {[p.name for p in PRESETS]} -- the glob is wrong"


@pytest.mark.parametrize("preset", PRESETS, ids=[p.name for p in PRESETS])
def test_every_preset_key_is_a_real_parameter(preset):
    known = _schema_params()
    data = json.loads(preset.read_text())
    unknown = sorted(k for k in data if not k.startswith("_comment") and k not in known)
    assert not unknown, (
        f"params/{preset.name} sets {unknown}, which nextflow_schema.json does not "
        "declare. A preset key the schema does not know configures nothing."
    )


@pytest.mark.parametrize("preset", PRESETS, ids=[p.name for p in PRESETS])
def test_a_preset_that_configures_a_backend_selects_it(preset):
    data = json.loads(preset.read_text())
    selected = data.get("seg_method")
    owners = {_owner(k) for k in data if not k.startswith("_comment")}
    owners.discard(None)
    stranded = sorted(o for o in owners if o != selected)
    assert not stranded, (
        f"params/{preset.name} sets parameters owned by {stranded} but its "
        f"seg_method is {selected!r}. Those values are silently discarded: the "
        "backend that reads them is never selected."
    )


@pytest.mark.parametrize("preset", PRESETS, ids=[p.name for p in PRESETS])
def test_a_preset_does_not_bake_in_site_sizing(preset):
    if preset.name in SITE_SIZING_EXEMPT:
        pytest.skip(SITE_SIZING_EXEMPT[preset.name])
    data = json.loads(preset.read_text())
    baked = sorted(k for k in SITE_SIZING_KEYS if k in data)
    assert not baked, (
        f"params/{preset.name} sets {baked}, which R4 reserves for "
        "conf/site.config.template. A -params-file value overrides -c, so these "
        "would silently defeat any site.config layered on top of this preset."
    )


def test_the_site_sizing_exemption_is_not_stale():
    """Every exempted preset still sets a site-sizing key, so no exemption is stale."""
    for name, reason in SITE_SIZING_EXEMPT.items():
        assert reason.strip(), f"{name} has an empty reason"
        preset = REPO / "params" / name
        assert preset.exists(), f"SITE_SIZING_EXEMPT names {name}, which does not exist"
        data = json.loads(preset.read_text())
        assert any(k in data for k in SITE_SIZING_KEYS), (
            f"SITE_SIZING_EXEMPT names {name}, which no longer sets any of "
            f"{SITE_SIZING_KEYS} -- the exemption is dead"
        )


def test_backend_prefixes_covers_exactly_the_schema_backends():
    """BACKEND_PREFIXES and the schema's seg_method enum name the same backends.

    A missing entry would exempt a backend's parameters from the rule above; a
    stale one would check a backend no preset can select."""
    assert set(BACKEND_PREFIXES) == _schema_seg_method_enum()


def test_the_owner_map_recognises_the_names_it_is_written_for():
    assert _owner("segmentation_model_dir") == "stardist"
    assert _owner("seg_instantseg_tile_size") == "instantseg"
    assert _owner("seg_cellsam_block_size") == "cellsam"
    # Shared across backends -- owned by none.
    assert _owner("seg_gpu") is None
    assert _owner("seg_expand_distance") is None
