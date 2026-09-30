#!/usr/bin/env python3
"""Guard against duplicated parameter defaults.

`nextflow.config`'s `params {}` block is the single source of truth for
parameter defaults. Process scripts and `conf/modules.config` must not
re-declare a *concrete* fallback via Groovy's `?:` operator, because `?:` is
falsy-coalescing (not null-coalescing): `params.tilex ?: 256` silently
rewrites a legitimate `--tilex 0` override to `256`, and if the fallback
literal ever drifts from `nextflow.config` the module lies about the real
default.

`params.x ?: <literal>` is only legitimate where `nextflow.config` declares
`x = null` — there, `?:` is the live "null means derive a value at runtime"
contract (e.g. `reg_jvm_heap_gb`, which falls back to a task.attempt-derived
heap size in modules/local/register.nf).

This test derives its allowlist directly from `nextflow.config`'s `params {}`
block, so it keeps working whether a param is null-declared today or someone
flips it to a concrete value tomorrow (or vice versa) -- no hand-curated list
to fall out of sync.

A second, independent check below closes the same blind spot on the Python
side: `bin/**/*.py` argparse defaults are just as capable of drifting from
`nextflow.config` as a Groovy `?:` fallback, and the pipeline never notices
because it always passes the flag explicitly -- only hand-invocation of the
script sees the stale default. That check is name-based (a flag normalizes
to a params key or it's out of scope -- no fuzzy/value matching) and parses
Python with `ast`, not regex, so a non-literal `default=` (a variable, a
call, an f-string) can be detected and skipped rather than mis-compared.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

from tests.nfmodel import block_extent as _block_extent
from tests.nfmodel import strip_comments as _strip_comments
from tests.nfmodel import strip_comments_and_strings as _strip_comments_and_strings
from tests.stare_shims import shims, source_of

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "nextflow.config"
BIN_DIR = ROOT / "bin"
MODULES_CONFIG_PATH = ROOT / "conf/modules.config"

# `params.<name> ?:` -- the pattern under audit.
FALLBACK_RE = re.compile(r"params\.([A-Za-z_][A-Za-z0-9_]*)\s*\?:")

# Files scanned for `params.x ?: <literal>` fallbacks.
#
# `lib/*.groovy` is in the list because parameter reads move there: SegBackends holds
# the per-backend defaults SEGMENT's script block used to inline, including
# `params.instanseg_model_dir ?: "$PWD/.instanseg_cache"`. A fallback that escaped the
# scan by being one directory over would be exactly the drift this test exists to catch.
SCANNED_GLOBS = [
    "main.nf",
    "modules/local/*.nf",
    "conf/*.config",
    "workflows/*.nf",
    "subworkflows/**/*.nf",
    "lib/*.groovy",
]


def _extract_params_block(config_text: str) -> str:
    """Return the raw text inside the top-level `params { ... }` block.

    The block's extent is found by `tests.nfmodel.block_extent`'s comment/
    string-aware walk rather than a naive brace count, so a `{`/`}` mentioned
    inside a comment or a quoted default value inside the block cannot
    mis-balance it and silently truncate (or overrun) what gets parsed below.
    """
    start = config_text.find("params {")
    if start == -1:
        raise ValueError("Could not locate `params { ... }` block in nextflow.config")

    brace_start = config_text.find("{", start)
    end = _block_extent(config_text, brace_start + 1)
    return config_text[brace_start + 1 : end - 1]


def parse_declared_params(config_text: str) -> dict[str, str]:
    """Parse the `params {}` block into {name: declared_value_text}.

    `declared_value_text` is the trimmed right-hand side of the assignment
    (comments stripped), so callers can test `value == "null"`. Comments are
    stripped via `tests.nfmodel.strip_comments` (view B), which keeps string
    contents verbatim -- some declared defaults are themselves quoted strings
    that may contain `//`, e.g. a URL -- and shares the same comment/string
    boundary walk as `_extract_params_block`'s `block_extent` above rather
    than re-deriving it with a private per-line quote-toggle that (unlike the
    shared walk) does not understand an escaped quote inside a string.
    """
    block = _extract_params_block(config_text)
    clean_block = _strip_comments(block)
    declared: dict[str, str] = {}
    for raw_line in clean_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if m:
            declared[m.group(1)] = m.group(2)
    return declared


def find_fallback_sites() -> list[tuple[Path, int, str, str]]:
    """Return (file, line_no, param_name, line_text) for every `params.x ?:` site.

    Scanned against `tests.nfmodel.strip_comments_and_strings` (view A), which
    blanks both comments and string contents while preserving every line
    number and character offset -- so a `params.x ?:` mention inside a `//`
    explanation or a quoted error message cannot be counted as a real
    fallback site. `line_text` in the returned tuple is still the ORIGINAL
    raw line, for a readable error message.
    """
    sites: list[tuple[Path, int, str, str]] = []
    for pattern in SCANNED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            text = path.read_text()
            raw_lines = text.splitlines()
            clean_lines = _strip_comments_and_strings(text).splitlines()
            for line_no, clean_line in enumerate(clean_lines, start=1):
                for m in FALLBACK_RE.finditer(clean_line):
                    sites.append(
                        (path, line_no, m.group(1), raw_lines[line_no - 1].strip())
                    )
    return sites


def test_no_duplicate_param_defaults():
    """Every `params.x ?: <literal>` site must name a param declared `null`.

    Where `nextflow.config` gives `x` a concrete default, any `?:` fallback
    beside `params.x` is unreachable duplication (or worse: a stale literal
    that disagrees with the real default) and must be deleted, leaving the
    bare `params.x`.
    """
    config_text = CONFIG_PATH.read_text()
    declared = parse_declared_params(config_text)
    sites = find_fallback_sites()

    assert sites, "No `params.x ?:` sites found -- scan patterns may be stale."

    offending = []
    for path, line_no, name, line_text in sites:
        if name not in declared:
            offending.append(
                f"{path.relative_to(ROOT)}:{line_no}: params.{name} ?: ... "
                f"(param not found in nextflow.config params{{}} block at all)"
            )
            continue
        if declared[name] != "null":
            offending.append(
                f"{path.relative_to(ROOT)}:{line_no}: params.{name} ?: ... "
                f"duplicates nextflow.config's concrete default "
                f"`{name} = {declared[name]}` -- delete the `?: ...` fallback."
            )

    assert not offending, (
        f"{len(offending)} site(s) duplicate a concrete nextflow.config default "
        "via `?:`. Delete the fallback literal, leaving the bare `params.x`, "
        "unless nextflow.config declares the param `null` (in which case `?:` "
        "is the live null-means-derive contract):\n" + "\n".join(offending)
    )


# ---------------------------------------------------------------------------
# bin/**/*.py argparse defaults vs. nextflow.config
# ---------------------------------------------------------------------------
#
# The `?:` check above only sees Groovy. A `bin/foo.py` argparse `default=`
# is just as capable of drifting from `nextflow.config` -- the pipeline
# always passes the flag explicitly (ext.args / a process script), so a
# stale Python default only bites hand-invocation of the script.
#
# The authority for "which param does this flag mean" is the pipeline's own
# invocation text, not a guess from the flag's spelling. `build_flag_param_maps`
# scans `modules/local/*.nf` + `conf/modules.config` for the two shapes the
# pipeline actually uses (`--flag ${params.key}` directly, or one level of
# `def var = params.key` aliasing) and derives TWO maps:
#
#   - a PER-SCRIPT map, keyed by `(invoking_script_basename, flag)`. This is
#     the primary authority: it disambiguates a flag that means different
#     params in different scripts, e.g. `--scale-factor` is `qc_scale_factor`
#     when `generate_registration_qc.py` is the invoking script but
#     `preprocess_qc_scale_factor` when it's `generate_preprocess_qc.py` --
#     genuinely unambiguous once you know which script is being invoked, even
#     though the bare flag name collides.
#   - a flag-only map (the same shape the original version of this check
#     shipped with), used as a fallback when no invoking script could be
#     identified for a given occurrence.
#
# A flag whose name happens to equal a params key (e.g. `--pyramid-resolutions`
# / `pyramid_resolutions`) is *also* covered by the direct form, so the
# straight name-equality check that shipped before this exists purely as the
# LAST-resort fallback, below both derived maps -- it's still correct when it
# fires, just not the primary authority. Full precedence: per-script map ->
# flag-only map -> name-equality. A flag like `bin/warp_seg_qc.py`'s
# `--method` is never sourced from `params.*`: `modules/local/warp_seg_qc.nf`
# dispatches it via `lib/WarpBackends.groovy`, keyed on the `method` INPUT --
# the valis backend omits the flag, relying on the Python default, and the
# tiled backend passes the literal `--method tiled` -- it stays out of scope
# by design, permanently, not just until someone wires it -- and is counted
# as such (see the `no_correspondence` bucket in
# `find_argparse_default_sites`), not silently dropped.
#
# (`segment_to_geojson.py`'s `--tolerance` was this section's example until
# Task 6 (arch-1-8) wired `--tolerance ${params.simplify_tolerance}` into
# `conf/modules.config`'s `SEG_QC_GEOJSON` ext.args -- it now resolves via the
# flag-only map into the `matched` bucket instead, which is exactly the
# mechanism this comment is illustrating. Swapped the example for one that
# cannot be resolved by any config wiring, so it does not rot the same way a
# second time.)

# Deliberate divergences between a bin/*.py argparse default and the
# nextflow.config default its flag resolves to (via the derived map or the
# name-equality fallback). Keyed by "<file>:<flag>". Every entry here is a
# documented, intentional standalone-use fallback (verified against the
# script's own help text/docstring) -- not a placeholder for a fix deferred
# out of scope. A divergence that is simply a bug gets fixed, not allowlisted
# (see bin/segment_cellsam.py's --block-size, which the derived map caught
# and was fixed rather than exempted).
ARGPARSE_DEFAULT_ALLOWLIST = {
    "seg_quality_eval.py:--max-pixels": {
        "reason": (
            "Matching the config default here would INVERT the meaning of "
            "cse_max_pixels = null. null means 'score at full resolution', and "
            "conf/modules.config expresses that by passing no --max-pixels flag "
            "at all (`params.cse_max_pixels ? \"--max-pixels ...\" : ''`). The "
            "script's own default must therefore be None = no cap; giving it "
            "50000000 would silently downsample exactly the runs that asked not "
            "to be downsampled, and the composite QualityScore shifts with the "
            "binning factor, so the corruption would be a plausible number "
            "rather than an error."
        ),
    },
    "extract_cell_properties.py:--outdir": {
        "reason": (
            "Not actually the same 'outdir' as nextflow.config's: "
            "modules/local/extract_cell_properties.nf:43 and "
            "modules/local/extract_nuclei_properties.nf:42 (which reuses this "
            "same script) both hardcode `--outdir .` -- a literal, not "
            "`${params.outdir}` -- because Nextflow already isolates each "
            "task in its own work dir; `publishDir` (driven by the real, "
            "pipeline-wide params.outdir) copies the results out afterward. "
            "This flag's name-equality match to the top-level `outdir` param "
            "is a coincidence, not a real correspondence -- the script's own "
            "default='.' (for standalone hand-invocation, writing to the "
            "cwd) is correct to keep as-is."
        ),
    },
    "quantify.py:--outdir": {
        "reason": (
            "Same non-correspondence as extract_cell_properties.py's --outdir: "
            "modules/local/quantify.nf:51 hardcodes `--outdir .` (a literal, "
            "not `${params.outdir}`) for the same task-local-vs-publishDir "
            "reason. The name-equality match to the top-level `outdir` param "
            "is coincidental."
        ),
    },
    "segment_instantseg.py:--pixel-size": {
        "reason": (
            "default=None is a real fallback, not a stale literal: the "
            "script's own help text says 'Override pixel size (um/px). If "
            "omitted, InstanSeg auto-detects from OME metadata.' The pipeline "
            "always passes an explicit value regardless (conf/modules.config's "
            "instantseg ext.args has `--pixel-size ${params.pixel_size}`), so "
            "this default only matters for standalone use, where auto-detect "
            "is the more useful behavior than silently assuming 0.325."
        ),
    },
    "apply_basic_profiles.py:--pixel-size": {
        "reason": (
            "nextflow.config's pixel_size default changed null -> 'auto' so an "
            "unresolvable default cannot reach a process (PREFLIGHT_SCALE catches it "
            "first). This script's own default=None is unaffected by that: "
            "modules/local/apply_profiles.nf:50 always passes `--pixel-size "
            "${params.pixel_size}` explicitly, so the argparse default only matters "
            "for standalone invocation, where resolve_pixel_size(None, ...) raising "
            "a clear 'you must pass a value' error is more useful than silently "
            "defaulting to metadata auto-detection."
        ),
    },
    "convert_image.py:--pixel_size": {
        "reason": (
            "Same contract as apply_basic_profiles.py's --pixel-size: "
            "modules/local/convert_image.nf:23,42 always builds `--pixel_size "
            "${pixel_size}` from params.pixel_size and passes it explicitly, so "
            "this script's own default=None -- unchanged by the config's null -> "
            "'auto' move -- only matters for standalone invocation."
        ),
    },
    "export_geojson.py:--pixel_size": {
        "reason": (
            "Not just unaffected by the config default but actively incompatible "
            "with it: EXPORT_GEOJSON works from a quantification CSV and contour "
            "JSON, not an image (see the script's own comment at its --pixel_size "
            "add_argument), so 'auto' has nothing to read a scale from and "
            "resolve_pixel_size() raises immediately if it is passed. "
            "modules/local/export_geojson.nf:45-48,62 documents that "
            "--pixel_size is 'NOT optional here' and always passes "
            "${params.pixel_size} explicitly (already resolved to a number by "
            "this point in the pipeline), so default=None -- 'you forgot to pass "
            "a number' -- is the only correct standalone default."
        ),
    },
    "export_spatialdata.py:--pixel-size": {
        "reason": (
            "Same contract as apply_basic_profiles.py's --pixel-size: "
            "modules/local/export_spatialdata.nf:61 always passes `--pixel-size "
            "${params.pixel_size}` explicitly, so this script's own default=None "
            "only matters for standalone invocation and is unaffected by the "
            "config's null -> 'auto' move."
        ),
    },
    "join_flowpath.py:--pixel-size": {
        "reason": (
            "No module invokes this script at all -- it is the standalone "
            "MIRAGE<->FlowPath join tool, run by hand outside the pipeline (see "
            "its module docstring). Its --pixel-size default therefore has no "
            "real correspondence to nextflow.config's pixel_size; the match is "
            "flag-name coincidence, same class of exemption as "
            "extract_cell_properties.py's --outdir above."
        ),
    },
    "merge_channels_pyramid.py:--physical-size-x": {
        "reason": (
            "Same contract as apply_basic_profiles.py's --pixel-size: "
            "modules/local/merge_and_pyramid.nf:29,57 always builds "
            "`--physical-size-x ${pixel_size_x}` from params.pixel_size and "
            "passes it explicitly, so this script's own default=None only "
            "matters for standalone invocation and is unaffected by the config's "
            "null -> 'auto' move."
        ),
    },
    "merge_channels_pyramid.py:--physical-size-y": {
        "reason": (
            "Same contract as --physical-size-x directly above: "
            "modules/local/merge_and_pyramid.nf:30,58 always builds "
            "`--physical-size-y ${pixel_size_y}` from params.pixel_size and "
            "passes it explicitly."
        ),
    },
    "split_multichannel.py:--pixel-size": {
        "reason": (
            "Same contract as apply_basic_profiles.py's --pixel-size: "
            "modules/local/split_channels.nf:73 always passes `--pixel-size "
            "${params.pixel_size}` explicitly, so this script's own default=None "
            "only matters for standalone invocation and is unaffected by the "
            "config's null -> 'auto' move."
        ),
    },
    "tiled_stitch.py:--pixel-size": {
        "reason": (
            "Same contract as apply_basic_profiles.py's --pixel-size: "
            "modules/local/tiled_stitch.nf:43 always passes `--pixel-size "
            "${params.pixel_size}` explicitly, so this script's own default=None "
            "only matters for standalone invocation and is unaffected by the "
            "config's null -> 'auto' move."
        ),
    },
    "generate_registration_qc.py:--nuclear-markers": {
        "reason": (
            "Same contract as merge_quant_csvs.py's and split_multichannel.py's "
            "--nuclear-markers: default=None is intentional per the script's own help "
            "text, 'Defaults to metadata.DEFAULT_NUCLEAR_MARKERS when omitted.' "
            "Confirmed true -- modules/local/generate_registration_qc.nf builds "
            "`--nuclear-markers ${MarkerUtils.markerList(params.nuclear_markers)"
            ".join(' ')}` unconditionally, so the pipeline never reaches the default. "
            "Mirroring ['DAPI','CELLTOX'] here would be a SECOND declaration of that "
            "default in Python; utils/metadata.py's DEFAULT_NUCLEAR_MARKERS is the one "
            "permitted mirror, and passing None is how this script defers to it."
        ),
    },
    "merge_quant_csvs.py:--nuclear-markers": {
        "reason": (
            "Same contract as tile_for_basic.py's and split_multichannel.py's "
            "--nuclear-markers: default=None is intentional per the script's own "
            "help text, 'MERGE_QUANT_CSVS passes params.nuclear_markers; the "
            "default is only for standalone use.' Confirmed true -- "
            "modules/local/merge_quant_csvs.nf builds the flag unconditionally from "
            "MarkerUtils.markerList, so the pipeline never reaches the default. "
            "Mirroring ['DAPI','CELLTOX'] here would be a SECOND declaration of "
            "that default in Python; utils/metadata.py's DEFAULT_NUCLEAR_MARKERS is "
            "the one permitted mirror, and passing None is how this script defers "
            "to it."
        ),
    },
    "tile_for_basic.py:--nuclear-markers": {
        "reason": (
            "Same contract as the deleted preprocess.py's --nuclear-markers, whose "
            "call site it replaced: default=None is intentional per the script's own "
            "help text, and "
            "modules/local/tile_for_basic.nf builds `--nuclear-markers "
            "${MarkerUtils.markerList(params.nuclear_markers).join(' ')}` "
            "unconditionally, so the pipeline never reaches the default. Mirroring "
            "['DAPI','CELLTOX'] in Python would be a SECOND declaration of that "
            "default; utils/metadata.py's DEFAULT_NUCLEAR_MARKERS is the one permitted "
            "mirror, and passing None is how this script defers to it."
        ),
    },
    "split_multichannel.py:--nuclear-markers": {
        "reason": (
            "Same pattern as segment_to_geojson.py, confirmed the same way: "
            "modules/local/split_channels.nf:40 always builds and passes "
            "`--nuclear-markers ${MarkerUtils.markerList(params.nuclear_"
            "markers).join(' ')}` -- 'SPLIT_CHANNELS always passes params."
            "nuclear_markers; the default is only for standalone use' per "
            "the script's own help text."
        ),
    },
}
# NOTE: `mask_to_geojson.py:--tolerance` is deliberately NOT here either. An earlier
# version of this allowlist exempted it as a "flag-name coincidence, not a real
# correspondence" -- wrong on both counts: it IS the same Douglas-Peucker quantity as
# segment_to_geojson.py's --tolerance, not a coincidence (segment_to_geojson.py:74-76
# passes its own --tolerance value straight into mask_to_feature_collection()'s
# simplify_tolerance), and the cited "+0.5 corner-of-pixel offset" justification
# (bin/mask_to_geojson.py:36-43, whose rule is owned by bin/utils/pixel_convention.py)
# is about contour OFFSET, an unrelated concern, not
# simplification tolerance -- it does not support keeping a divergent default. Per this
# file's own policy ("a divergence that is simply a bug gets fixed, not allowlisted"),
# bin/mask_to_geojson.py's --tolerance default was changed 0.5 -> 1.0 to match its
# sibling instead of being exempted.
# NOTE: `segment.py:--model-name` and `segment_to_geojson.py:--model-name`
# are deliberately NOT here. Both are `required=True` with no `default=` at
# all, so their comparison against `segmentation_model` was a moot,
# unreachable-default artifact, not a real divergence -- `find_argparse_
# default_sites` now excludes every `required=True` add_argument() call from
# comparison entirely (see the `required` bucket below), so these two simply
# never reach a point where they'd need an exemption.


def _config_value_to_python(text: str):
    """Translate a `nextflow.config` params{} RHS into a comparable Python value.

    Groovy's `true`/`false`/`null` aren't Python literals; quoted strings,
    numbers, and list literals of those already parse as valid Python via
    `ast.literal_eval` (Groovy's `['a', 'b']` is also a valid Python list
    literal).
    """
    keyword_literals = {"true": True, "false": False, "null": None}
    if text in keyword_literals:
        return keyword_literals[text]
    return ast.literal_eval(text)


def _normalize_flag(flag: str) -> str:
    """`--foo-bar` / `--foo_bar` (or a short `-x`) -> `foo_bar` / `x`."""
    return flag.lstrip("-").replace("-", "_")


# A CLI flag immediately followed by one-or-more consecutive `${...}`
# interpolations, e.g. `--fov_size ${params.preproc_tile_size}` or
# `--n-tiles ${params.seg_n_tiles_y} ${params.seg_n_tiles_x}` (two
# interpolations -- a composite flag this check deliberately does not
# resolve; see `build_flag_param_maps`).
_FLAG_INTERP_RE = re.compile(r"--([A-Za-z0-9][A-Za-z0-9_-]*)((?:\s+\$\{[^}]*\})+)")
# A single `${...}` interpolation's content, only when it is one bare token
# (a dotted `params.key` reference or a plain variable name) with no
# surrounding whitespace -- i.e. not `${params.x ?: task.cpus}` or
# `${row.ix}`-via-method-call style expressions, which are one level deeper
# than this check resolves.
_INTERP_ATOM_RE = re.compile(r"\$\{\s*([^}\s]+)\s*\}")
# `def <var> = params.<key>` -- the one level of aliasing this check resolves.
_DEF_ALIAS_RE = re.compile(
    r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*params\.([A-Za-z_][A-Za-z0-9_]*)\s*$",
    re.MULTILINE,
)
# A line in a `modules/local/*.nf` `script:` block consisting solely of the
# invoked bin/*.py script's name (Nextflow's `stripIndent()`'d triple-quoted
# strings escape a trailing line-continuation as one or two literal `\`
# characters, hence `\\*` rather than `\\?`). This is how this check learns
# "which script is this text talking about" -- e.g. `tile_for_basic.py \\` in
# modules/local/tile_for_basic.nf.
_SCRIPT_LINE_RE = re.compile(
    r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*\.py)[ \t]*\\*[ \t]*$", re.MULTILINE
)
# A backend-table entry in a `lib/*.groovy` file shaped like SegBackends.groovy's
# `stardist: [container: '...', entrypoint: 'segment.py', ...]`: a bare
# identifier key, immediately opening a list whose first two fields are
# `container:` then `entrypoint: '<script>.py'`. This is how this check
# resolves conf/modules.config's SEGMENT `ext.args`, which dispatches THREE
# different bin/*.py scripts through a `params.seg_method`-keyed map of flag
# lists but (unlike every other process in conf/modules.config) never spells
# any of those three scripts' names in that map -- the entrypoint->script
# correspondence lives only in this backend table. Keys are discovered
# generically (whatever backends the table declares), not hardcoded to
# 'stardist'/'instantseg'/'cellsam', so a future backend needs no code change
# here to be picked up.
_ENTRYPOINT_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[\s*container\s*:.*?entrypoint\s*:\s*'([\w.]+\.py)'",
    re.DOTALL,
)


def _extract_atoms(
    text: str, aliases: dict[str, str]
) -> tuple[list[tuple[str, str]], set[str]]:
    """Find `--flag ${...}` occurrences in `text`.

    Returns `(resolved, multivalue)`: `resolved` is `(flag, param_key)` for
    every occurrence resolving to exactly one bare `params.key` or aliased
    variable; `multivalue` is the set of flag names seen with more than one
    interpolation (a composite/nargs-style flag -- see `build_flag_param_maps`).
    """
    resolved: list[tuple[str, str]] = []
    multivalue: set[str] = set()
    for m in _FLAG_INTERP_RE.finditer(text):
        flag = _normalize_flag(m.group(1))
        atoms = _INTERP_ATOM_RE.findall(m.group(2))
        if len(atoms) != 1:
            multivalue.add(flag)
            continue
        atom = atoms[0]
        if atom.startswith("params."):
            resolved.append((flag, atom[len("params.") :].split(".")[0]))
        elif atom in aliases:
            resolved.append((flag, aliases[atom]))
        # else: not a bare params ref and not a known simple alias -- a `?:`
        # fallback or a method call -- one level too deep, left unresolved.
    return resolved, multivalue


def build_flag_param_maps() -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Derive authoritative flag->param correspondences from how the
    pipeline itself invokes bin/*.py scripts, instead of guessing from the
    flag's spelling.

    Scans `modules/local/*.nf` and `conf/modules.config` for exactly two
    shapes (this is deliberately NOT a general expression evaluator -- it
    resolves at most one level):

      - direct:      `--flag ${params.key}`
      - def-aliased: `def var = params.key`  ...  `--flag ${var}`

    Returns `(per_script, flag_only)`:

      - `per_script`: keyed by `(invoking_script_basename, flag)`. This is
        the primary authority, because it disambiguates flags that mean
        different params in different scripts -- e.g. `--model-name` is
        `segmentation_model` when `segment.py` is the invoking script, but
        `seg_instantseg_model` when it's `segment_instantseg.py`. The invoking
        script for a `modules/local/*.nf` file's flags is read from the
        nearest preceding `_SCRIPT_LINE_RE` match in that file (the file is
        split into per-script segments, so a file invoking more than one
        script, like register.nf's register.py then
        create_channels_manifest.py, attributes each flag to the right one).
        For `conf/modules.config`'s SEGMENT `ext.args` -- the one process
        that dispatches three different scripts through a
        `params.seg_method`-keyed flag-list map, without ever naming any of
        them in that map -- the invoking script per backend key is resolved
        via `_ENTRYPOINT_RE` against `lib/*.groovy`'s backend table (the only
        place that correspondence is actually declared).
      - `flag_only`: the flag-name-only map (no script disambiguation),
        aggregated across every occurrence everywhere (including ones
        already captured in `per_script`). Serves as the fallback when a
        flag's invoking script couldn't be identified for a given
        occurrence.

    A (script, flag) pair -- or, for `flag_only`, a bare flag -- is excluded
    from its map when:

      - it resolves to more than one *distinct* param key within that same
        scope. For `flag_only` this is common and expected (that's exactly
        why `per_script` exists: e.g. bare `--model-name` is ambiguous
        between `segmentation_model` and `seg_instantseg_model`, but
        `("segment.py", "model_name")` and `("segment_instantseg.py",
        "model_name")` are each unambiguous). Within `per_script`, a
        collision would mean the same script's own flag maps to two
        different params, which never happens in this codebase but would
        indicate a scan bug if it did.
      - any occurrence of the flag is followed by more than one
        interpolation -- a composite/nargs-style flag consuming multiple
        params at once, e.g. `--n-tiles ${params.seg_n_tiles_y}
        ${params.seg_n_tiles_x}` (comparing a 2-element argparse default
        against one scalar param would be a category error, not a real
        drift check); or
      - the interpolated expression isn't a bare `params.key` or a bare
        aliased variable -- e.g. an expression like `${params.some_knob ?:
        task.cpus}` is a fallback one level deeper than this resolves, so
        its flag is intentionally left unmapped rather than compared
        against a null default (which would be a false positive: null
        there means "derive at runtime", the exact contract
        `test_no_duplicate_param_defaults` already recognizes for the
        Groovy `?:` check above).
    """
    per_script_raw: dict[tuple[str, str], set[str]] = {}
    flag_only_raw: dict[str, set[str]] = {}
    multivalue: set[str] = set()

    def record(script: str | None, flag: str, key: str) -> None:
        if script is not None:
            per_script_raw.setdefault((script, flag), set()).add(key)
        flag_only_raw.setdefault(flag, set()).add(key)

    for path in sorted(ROOT.glob("modules/local/*.nf")):
        # `tests.nfmodel.strip_comments` (view B) blanks comments only, so a
        # `--flag ${params.x}` or `def var = params.x` mention inside a `//`
        # explanation cannot be picked up here -- while a triple-quoted
        # script: body's own invocation lines (the actual match target for
        # _FLAG_INTERP_RE/_SCRIPT_LINE_RE) survive untouched, unlike view A
        # (strip_comments_and_strings), which would blank that string's
        # contents along with the flags this function exists to find.
        text = _strip_comments(path.read_text())
        aliases = dict(_DEF_ALIAS_RE.findall(text))
        script_matches = list(_SCRIPT_LINE_RE.finditer(text))
        # Text before the first script-name line (rare, but flags here can't
        # be attributed to any particular script).
        segments: list[tuple[str | None, str]] = [
            (None, text[: script_matches[0].start()] if script_matches else text)
        ]
        for i, sm in enumerate(script_matches):
            end = (
                script_matches[i + 1].start()
                if i + 1 < len(script_matches)
                else len(text)
            )
            segments.append((sm.group(1), text[sm.end() : end]))
        for script, segment_text in segments:
            resolved, mv = _extract_atoms(segment_text, aliases)
            multivalue |= mv
            for flag, key in resolved:
                record(script, flag, key)

    config_text = _strip_comments(MODULES_CONFIG_PATH.read_text())
    backend_script: dict[str, str] = {}
    for lib_path in sorted(ROOT.glob("lib/*.groovy")):
        for m in _ENTRYPOINT_RE.finditer(_strip_comments(lib_path.read_text())):
            backend_script[m.group(1)] = m.group(2)

    backend_spans: list[tuple[int, int]] = []
    if backend_script:
        alternation = "|".join(re.escape(key) for key in backend_script)
        backend_block_re = re.compile(rf"({alternation})\s*:\s*\[(.*?)\]", re.DOTALL)
        for m in backend_block_re.finditer(config_text):
            backend_spans.append((m.start(), m.end()))
            resolved, mv = _extract_atoms(m.group(2), {})
            multivalue |= mv
            for flag, key in resolved:
                record(backend_script.get(m.group(1)), flag, key)

    # Mask out the backend blocks already handled above so the remaining,
    # non-backend-dispatched ext.args entries are only counted once (into
    # flag_only, since conf/modules.config's other withName blocks don't
    # multiplex several scripts through one block the way SEGMENT does).
    remainder = list(config_text)
    for start, end in backend_spans:
        for i in range(start, end):
            remainder[i] = " "
    resolved, mv = _extract_atoms("".join(remainder), {})
    multivalue |= mv
    for flag, key in resolved:
        record(None, flag, key)

    per_script = {k: next(iter(v)) for k, v in per_script_raw.items() if len(v) == 1}
    flag_only = {
        f: next(iter(v))
        for f, v in flag_only_raw.items()
        if len(v) == 1 and f not in multivalue
    }
    return per_script, flag_only


def _iter_add_argument_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            yield node


def _is_required_true(call: ast.Call) -> bool:
    """True iff this `add_argument()` call has a literal `required=True`.

    A `required=True` argument can never fall back to its `default=` --
    argparse raises if the flag is omitted, so nothing can silently drift to
    an unreachable default. This is true whether or not the call even has a
    `default=` kwarg: an argument that is BOTH `required=True` AND carries a
    `default=` is unusual but legal, and that default is dead code -- it can
    never execute -- for exactly the same reason, so it is excluded from
    comparison identically to the more common required-with-no-default case.
    A non-literal `required=` (a variable, an expression) is treated as NOT
    required, matching this file's general policy of only acting on what can
    be statically confirmed.
    """
    for kw in call.keywords:
        if kw.arg == "required":
            try:
                return ast.literal_eval(kw.value) is True
            except (ValueError, TypeError):
                return False
    return False


def _shim_name(source: Path) -> str:
    """The bin/ file name a scanned source belongs to: itself, or the shim over it."""
    for shim, target in shims().items():
        if target == source:
            return shim.name
    return source.name


def find_argparse_default_sites():
    """Scan `bin/**/*.py` for `add_argument()` calls whose flag resolves to a
    `nextflow.config` params{} key, via the per-script map, the flag-only
    map, or (as a last resort) exact name-equality.

    Returns `(matched, skipped, required, no_correspondence, declared)`:
      - `matched`: `(path, flag, param_name, python_default, via)` for calls
        with a literal `default=` (or no `default=` kwarg at all, which
        argparse itself treats as `default=None`), where the argument is
        NOT `required=True`. `via` is `"per-script"`, `"flag-only"`, or
        `"name-eq"`, per the resolution precedence.
      - `skipped`: `(path, flag, param_name, via)` for calls whose
        `default=` is a non-literal expression (a variable, a call, an
        f-string, ...) that `ast.literal_eval` cannot statically evaluate.
      - `required`: `(path, flag, param_name, via)` for calls with a literal
        `required=True` -- excluded from comparison regardless of whether a
        `default=` is present, because a required argument's default (if
        any) is dead code that can never be reached (see
        `_is_required_true`).
      - `no_correspondence`: `(path, flag)` for flags that resolved via
        none of the three paths -- genuinely out of scope (e.g.
        `--tolerance`), not a silently dropped result.
      - `declared`: the `nextflow.config` params{} block, as parsed by
        `parse_declared_params`.

    Every argparse flag this function examines lands in exactly one of
    `matched`, `skipped`, `required`, or `no_correspondence`, so their
    combined length accounts for all of them.
    """
    declared = parse_declared_params(CONFIG_PATH.read_text())
    per_script_map, flag_only_map = build_flag_param_maps()

    matched: list[tuple[Path, str, str, object, str]] = []
    skipped: list[tuple[Path, str, str, str]] = []
    required: list[tuple[Path, str, str, str]] = []
    no_correspondence: list[tuple[Path, str]] = []
    for shim in sorted(BIN_DIR.rglob("*.py")):
        # A STARE shim (bin/tiled_stitch.py etc.) carries no argparse of its own: its
        # flags live in the packages/stare stage it re-exports. Read that file, but keep
        # the SHIM's name for the per-script map -- that is the name the module invokes.
        path = source_of(shim)
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for call in _iter_add_argument_calls(tree):
            flags = [
                arg.value
                for arg in call.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("-")
            ]
            default_kw = next(
                (kw.value for kw in call.keywords if kw.arg == "default"), None
            )
            is_required = _is_required_true(call)
            for flag in flags:
                name = _normalize_flag(flag)
                if (shim.name, name) in per_script_map:
                    key, via = per_script_map[(shim.name, name)], "per-script"
                elif name in flag_only_map:
                    key, via = flag_only_map[name], "flag-only"
                elif name in declared:
                    key, via = name, "name-eq"
                else:
                    no_correspondence.append((path, flag))
                    continue
                if is_required:
                    required.append((path, flag, key, via))
                    continue
                if default_kw is None:
                    matched.append((path, flag, key, None, via))
                    continue
                try:
                    python_default = ast.literal_eval(default_kw)
                except (ValueError, TypeError):
                    skipped.append((path, flag, key, via))
                    continue
                matched.append((path, flag, key, python_default, via))
    return matched, skipped, required, no_correspondence, declared


def test_no_duplicate_bin_argparse_defaults():
    """Every `bin/**/*.py` argparse default that resolves to a
    `nextflow.config` param (via the per-script map, the flag-only map, or
    exact name-equality as a last resort) must equal that param's default --
    unless the argument is `required=True`, in which case its default (if
    it even has one) is dead code that can never execute, so it is excluded
    from comparison rather than compared against a value it can never
    actually take.

    Non-literal defaults (a variable, a call, an f-string) can't be
    statically compared and are skipped. Flags with no correspondence at all
    are genuinely out of scope. All three of these, plus the required-and-
    excluded case, are counted, not silently dropped -- the warning this
    test always emits reports FIVE numbers (checked via either derived map;
    checked via name-equality; skipped as non-literal; excluded as
    required=True; no correspondence found) whose sum is every argparse flag
    this function examined, so no flag can fall out of the accounting
    unobserved.
    """
    matched, skipped, required, no_correspondence, declared = (
        find_argparse_default_sites()
    )
    total_examined = (
        len(matched) + len(skipped) + len(required) + len(no_correspondence)
    )

    assert total_examined, (
        "No bin/**/*.py argparse add_argument() call was found at all -- "
        "the scan may be broken."
    )

    offending = []
    for path, flag, key, python_default, via in matched:
        # keyed by the SHIM's name for a STARE stage (`tiled_stitch.py:--pixel-size`),
        # the same name the module invokes and the per-script map is built from
        allowlist_key = f"{_shim_name(path)}:{flag}"
        if allowlist_key in ARGPARSE_DEFAULT_ALLOWLIST:
            continue
        config_default = _config_value_to_python(declared[key])
        if python_default != config_default:
            offending.append(
                f"{path.relative_to(ROOT)}: {flag} default={python_default!r} "
                f"but nextflow.config's {key} = {config_default!r} (resolved "
                f"via {via}). Either fix the Python default to match, or add "
                f"an ARGPARSE_DEFAULT_ALLOWLIST entry keyed {allowlist_key!r} "
                "with a reason."
            )

    via_per_script = sum(1 for *_, via in matched if via == "per-script")
    via_flag_only = sum(1 for *_, via in matched if via == "flag-only")
    via_map = via_per_script + via_flag_only
    via_name_eq = sum(1 for *_, via in matched if via == "name-eq")
    warnings.warn(
        f"bin argparse-default check examined {total_examined} flag(s) total: "
        f"{via_map} checked via the derived flag->param map "
        f"({via_per_script} per-script, {via_flag_only} flag-only), "
        f"{via_name_eq} via the name-equality fallback, {len(skipped)} "
        "skipped (non-literal default=, cannot be statically compared), "
        f"{len(required)} excluded as required=True (default, if any, is "
        f"dead code), {len(no_correspondence)} with no correspondence found "
        f"at all (out of scope). {via_map} + {via_name_eq} + {len(skipped)} "
        f"+ {len(required)} + {len(no_correspondence)} = {total_examined}."
        + (
            " Skipped: "
            + ", ".join(f"{p.relative_to(ROOT)}:{f}" for p, f, _, _ in skipped)
            if skipped
            else ""
        ),
        stacklevel=1,
    )

    assert not offending, (
        f"{len(offending)} bin/*.py argparse default(s) drifted from "
        "nextflow.config:\n" + "\n".join(offending)
    )


def test_argparse_default_allowlist_entries_have_reasons():
    """Every `ARGPARSE_DEFAULT_ALLOWLIST` entry must be a `{reason}` dict.

    Forces a non-empty, structured reason instead of a bare string or a
    silently-forgotten entry -- mirrors the shape enforced on
    `test_no_dead_bin_modules.py`'s `ALLOWLIST`.
    """
    for key, entry in ARGPARSE_DEFAULT_ALLOWLIST.items():
        assert isinstance(entry, dict) and set(entry) == {"reason"}, (
            f"ARGPARSE_DEFAULT_ALLOWLIST[{key!r}] must be a dict with exactly "
            "a 'reason' key."
        )
        assert isinstance(entry["reason"], str) and entry["reason"].strip(), (
            f"ARGPARSE_DEFAULT_ALLOWLIST[{key!r}]['reason'] must be a non-empty string."
        )


def test_seg_qc_pairing_and_radius_factor_match_bin_defaults():
    """Close the one blind spot `test_no_duplicate_bin_argparse_defaults` warns about
    but cannot check.

    `warp_seg_qc.py`'s `--pairing` and `--match-radius-factor` resolve (via
    `build_flag_param_maps`) to `nextflow.config`'s `seg_qc_pairing` and
    `seg_qc_match_radius_factor`, but their argparse `default=` is a NAME
    (`cp.DEFAULT_PAIRING`, `DEFAULT_MATCH_RADIUS_FACTOR`), not a literal --
    `ast.literal_eval` can't evaluate a name, so `find_argparse_default_sites()`
    files both under `skipped` and the comparison in
    `test_no_duplicate_bin_argparse_defaults` silently never runs for them (that
    test's own warning message says so under "skipped"). Nothing today would
    fail if `nextflow.config` said `seg_qc_match_radius_factor = 1.2` while
    `bin/warp_seg_qc.py`'s `DEFAULT_MATCH_RADIUS_FACTOR` stayed `1.5`.

    This test runs the actual Python constants (rather than statically parsing
    them) and compares them directly against `nextflow.config`'s declared
    defaults, closing the gap in both directions.
    """
    import sys

    bin_dir = str(BIN_DIR)
    bin_utils_dir = str(BIN_DIR / "utils")
    added = [p for p in (bin_dir, bin_utils_dir) if p not in sys.path]
    for p in added:
        sys.path.insert(0, p)
    try:
        import warp_seg_qc as wsq
        from utils import cell_pairs as cp
    finally:
        for p in added:
            sys.path.remove(p)

    declared = parse_declared_params(CONFIG_PATH.read_text())

    config_pairing = _config_value_to_python(declared["seg_qc_pairing"])
    assert config_pairing == cp.DEFAULT_PAIRING, (
        f"nextflow.config's seg_qc_pairing = {config_pairing!r} but "
        f"bin/utils/cell_pairs.py's DEFAULT_PAIRING = {cp.DEFAULT_PAIRING!r} "
        "-- these must match (warp_seg_qc.py's --pairing default is "
        "cp.DEFAULT_PAIRING)."
    )

    config_radius_factor = _config_value_to_python(
        declared["seg_qc_match_radius_factor"]
    )
    assert config_radius_factor == wsq.DEFAULT_MATCH_RADIUS_FACTOR, (
        "nextflow.config's seg_qc_match_radius_factor = "
        f"{config_radius_factor!r} but bin/warp_seg_qc.py's "
        f"DEFAULT_MATCH_RADIUS_FACTOR = {wsq.DEFAULT_MATCH_RADIUS_FACTOR!r} "
        "-- these must match (warp_seg_qc.py's --match-radius-factor default "
        "is this constant)."
    )
