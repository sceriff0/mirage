"""Guard: no process may render `params.pixel_size` directly unless it is handed an
image it can resolve `'auto'` from itself.

The CRITICAL defect this guards against: `nextflow.config` ships `pixel_size = 'auto'`,
a string nothing resolves except `PREFLIGHT_SCALE` (subworkflows/local/input_check.nf),
which reads every input image's OME metadata once and injects the resolved NUMBER into
`meta.pixel_size`. A process with no image of its own -- EXPORT_GEOJSON (a CSV),
EXPORT_SPATIALDATA (no pyramid unless `--spatialdata_include_image`), SEG_QUALITY_EVAL
(its own tool never reads scale from `image`, see its script comment), and SEGMENT's
instantseg backend (whose `--pixel-size` is `type=float`, so the literal string 'auto'
fails argparse before the tool ever runs) -- cannot resolve that string itself. Reading
`params.pixel_size` directly there reaches the tool as the literal 'auto' at the shipped
default; reading `meta.pixel_size` gets the number PREFLIGHT_SCALE already resolved.

This is a STATIC guard over the same tests/nfmodel this repo's other Nextflow-source
guards use (see tests/test_nfmodel.py's discovery rule) -- never a private regex over
`.nf`/`conf/modules.config` text.
"""

from __future__ import annotations

from tests.nfmodel import (
    param_refs,
    processes,
    strip_comments,
    with_name_blocks,
)

# Every entry here is a process that IS handed an image (or, for PREFLIGHT_SCALE, is
# THE resolver) and can therefore legitimately read params.pixel_size directly --
# resolve_pixel_size(params.pixel_size, <that image>, ...) is how each of bin/
# convert_image.py, bin/tiled_stitch.py, bin/apply_basic_profiles.py,
# bin/split_multichannel.py and bin/merge_channels_pyramid.py resolve 'auto' on their
# own (see tests/test_pixel_size_is_passed.py). A process reached ONLY via this file's
# guard failing is not automatically safe to add here -- add it only once you have
# confirmed its script is actually given an image and calls resolve_pixel_size (or
# equivalent) against it.
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
    """A safelist entry for a renamed/deleted process is dead weight that can hide a
    real regression (an entry that no longer matches anything makes the scan look
    stricter than it is). Every name must resolve to an actual process."""
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
