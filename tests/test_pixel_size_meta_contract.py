"""Guard: no process may render `params.pixel_size` directly unless it is handed an
image it can resolve `'auto'` from itself.

`nextflow.config` ships `pixel_size = 'auto'`, which only `PREFLIGHT_SCALE`
(subworkflows/local/input_check.nf) resolves, injecting the number into
`meta.pixel_size`. A process with no image of its own (EXPORT_GEOJSON,
EXPORT_SPATIALDATA, SEG_QUALITY_EVAL, SEGMENT's instantseg backend) would pass the
literal 'auto' to its tool; it must read `meta.pixel_size` instead.

A static guard built on tests/nfmodel.
"""

from __future__ import annotations

from tests.nfmodel import (
    param_refs,
    processes,
    strip_comments,
    with_name_blocks,
)

# Processes handed an image (or, for PREFLIGHT_SCALE, the resolver itself), whose
# scripts resolve 'auto' from that image via resolve_pixel_size. Add one only after
# confirming its script does that.
SAFE_PIXEL_SIZE_CONSUMERS = {
    "PREFLIGHT_SCALE",  # the one place params.pixel_size is resolved to a number
    "CONVERT_IMAGE",  # given the raw slide; resolves 'auto' from its own OME metadata
    "TILED_STITCH",  # given the moving image; resolves 'auto' itself
    "TILED_COARSE",  # given reference + moving; STARE reads 'auto' from their OME header
    "TILED_REG_TILE",  # given reference + moving; STARE reads 'auto' from their OME header
    "APPLY_PROFILES",  # given --image; resolves 'auto' itself
    "SPLIT_CHANNELS",  # given the registered image; resolves 'auto' itself
    "MERGE_AND_PYRAMID",  # given a probe image; resolves 'auto' itself
}


def _flags_pixel_size(text: str) -> bool:
    """True if the comment-stripped text reads `params.pixel_size`."""
    return "pixel_size" in param_refs(strip_comments(text))


def _module_offenders() -> list[str]:
    offenders = []
    for name, proc in processes().items():
        if name in SAFE_PIXEL_SIZE_CONSUMERS:
            continue
        text = proc.script_body + "\n" + proc.stub_body
        if _flags_pixel_size(text):
            offenders.append(f"modules/local process {name} ({proc.path.name})")
    return offenders


def _config_offenders() -> list[str]:
    offenders = []
    for block in with_name_blocks():
        if not _flags_pixel_size(block.raw_body):
            continue
        for name in block.names:
            if name not in SAFE_PIXEL_SIZE_CONSUMERS:
                offenders.append(
                    f"conf/modules.config withName: '{block.selector}' (as '{name}')"
                )
    return offenders


def test_the_safelist_still_names_real_processes():
    """Every safelist name resolves to an actual process, so no entry is stale."""
    known = set(processes())
    for block in with_name_blocks():
        known.update(block.names)
    missing = SAFE_PIXEL_SIZE_CONSUMERS - known
    assert not missing, (
        f"SAFE_PIXEL_SIZE_CONSUMERS names process(es) that do not exist: {missing}"
    )


def test_no_process_renders_params_pixel_size_directly_unless_safelisted():
    offenders = _module_offenders() + _config_offenders()
    assert not offenders, (
        "these consumers read params.pixel_size directly, which is the literal string "
        "'auto' at the shipped default (nextflow.config) and cannot be resolved without "
        "an image of their own. Use meta.pixel_size instead -- INPUT_CHECK's "
        "PREFLIGHT_SCALE already resolved it per-slide (subworkflows/local/input_check.nf) "
        "-- or, if this process really is handed an image and resolves 'auto' itself, add "
        "it to SAFE_PIXEL_SIZE_CONSUMERS with a reason:\n" + "\n".join(offenders)
    )
