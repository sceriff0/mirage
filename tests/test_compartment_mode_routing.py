#!/usr/bin/env python3
"""Static guard: nothing reads `params.quantify_compartments` /
`params.expanded_quantification` / `params.embed_masks` raw except the resolver
(`lib/ParamUtils.groovy`'s `compartmentMode()`) and a small, explicit allowlist
of config/module sites that legitimately keep their own read.

All THREE params are in scope, not just `quantify_compartments`: they are the
three fields `ParamUtils.compartmentMode()` resolves together into one map, and
`assemble_export.nf:78`'s original defect --
`params.embed_masks && params.quantify_compartments && params.expanded_quantification`
-- read all three raw in the SAME expression. A guard that only watched
`quantify_compartments` would pass while that exact line regressed, having
caught none of the other two names. (Confirmed directly: a planted
`params.embed_masks && params.expanded_quantification` read in
assemble_export.nf passed an earlier, narrower version of this guard 2/2 green.)

`quantify_compartments` used to be read directly at every consumer across
7+ files, including one ternary
(`ch_nuc_contours_for_export = params.quantify_compartments ? ... : ...`)
copied VERBATIM between postprocess.nf and add_cycle.nf, plus
assemble_export.nf's `embed_masks` gate reading all three related params raw.
Its two sibling routing params both already have a seam: --registration_method
is read ONCE (subworkflows/local/registration.nf) and handed down as an
argument; --nuclear_markers is routed through MarkerUtils.markerList() (see
test_nuclear_marker_routing.py). This test is the equivalent guard for the
compartment-quantification trio: `ParamUtils.compartmentMode(params)` is
resolved once in workflows/mirage.nf and threaded down as an argument to every
subworkflow that used to re-derive any of the three flags itself --
subworkflows/local/segmentation.nf's SEGMENTATION and
READ_SEGMENTED_CHECKPOINT, subworkflows/local/postprocess.nf,
subworkflows/local/add_cycle.nf, subworkflows/local/assemble_export.nf.

Two kinds of exception are legitimate and allowlisted below, each for a
DIFFERENT, precisely-stated reason -- conflating them is exactly the shape of
allowlist entry this repo has shipped before with a stated reason that had
counterexamples (see CLAUDE.md's verification-reality note on
`test_resource_label_coverage.py`):

  1. `modules/local/*.nf` `script:`/`stub:` blocks reading a param directly to
     build their own CLI flag string -- the established, accepted pattern in
     this repo (see CLAUDE.md: "modules/local/*.nf reading a param in a
     script: block to build its own flags is the established pattern"). Each
     such site is a leaf: it consumes the flag to render a command string and
     never re-exports the decision to another file, so no seam threading is
     possible there and none is owed.

  2. `conf/modules.config` reading `params.expanded_quantification` directly
     at line 645 (`ext.args = { params.expanded_quantification ? '--expanded'
     : '' }`) -- genuinely unavoidable: `conf/*.config` closures cannot see
     `lib/*.groovy` classes (a class name referenced there resolves against
     ConfigObject and fails only when the closure runs -- see CLAUDE.md's
     "Config can't see lib/ classes"), so `ext.args` MUST read params raw.
     This reason is checked, not assumed, and does NOT extend to
     `params.quantify_compartments` or `params.embed_masks`: neither is read
     anywhere in `conf/*.config` today (only a comment mentions the first
     name), so allowlisting the whole file "because config always gets a
     pass" would hide a real future regression of either. The allowlist entry
     below names the one line, not the file wholesale.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every place Nextflow/Groovy code can live -- identical to
# test_nuclear_marker_routing.py's SCANNED_GLOBS. `nextflow.config` is excluded:
# it DECLARES the parameters (the only place a default may live), it does not
# consume them. `tests/` is excluded: tests may reference the params freely
# (e.g. tests/lib_probe.nf's ParamUtils.compartmentMode() assertions) without
# being a production consumer.
SCANNED_GLOBS = [
    "main.nf",
    "modules/local/*.nf",
    "subworkflows/**/*.nf",
    "workflows/*.nf",
    "lib/*.groovy",
    "conf/*.config",
]

# The resolver itself. Its body is what "routed" means; it is not a consumer.
ROUTER_FILE = "ParamUtils.groovy"

# All three fields ParamUtils.compartmentMode() resolves together -- see the
# module docstring for why watching only one of the three is not enough.
READ_RE = re.compile(
    r"(?:\w+\.)?params\.(?:quantify_compartments|expanded_quantification|embed_masks)"
)

# path-relative-to-ROOT -> reason. A WHOLE-FILE exemption: every raw read in
# this file is covered, because the reason applies uniformly to the file
# (each is its own script:/stub: leaf building its own CLI flag). A new file
# appearing here is a regression this test exists to catch, not an invitation
# to widen this dict quietly.
ALLOWED_FILES = {
    "modules/local/quantify.nf": (
        "script: block -- builds its own --nuclei_mask_file flag from "
        "params.quantify_compartments, the same way every other "
        "ext.args-shaped flag this process builds is assembled inside its "
        "own script: block."
    ),
    "modules/local/export_geojson.nf": (
        "script:/stub: blocks -- script: builds its own "
        "--nucleus_contours_json flag from params.quantify_compartments; "
        "stub: mirrors that same branch (only touches "
        "cells_wholecell.geojson when the real run would have produced it) "
        "so the -stub output tree matches the real one."
    ),
    "modules/local/export_spatialdata.nf": (
        "script: block -- builds its own --nucleus-contours-json flag from "
        "params.quantify_compartments, the same leaf pattern as "
        "export_geojson.nf."
    ),
}

# path-relative-to-ROOT -> {1-based line numbers}. An EXACT-LINE exemption,
# stricter than ALLOWED_FILES: conf/modules.config's reason (ext.args cannot
# see lib/*.groovy classes) applies to the ONE line that actually needs it,
# not the file as a whole -- conf/modules.config does not read
# params.quantify_compartments or params.embed_masks anywhere today, and a
# whole-file exemption would hide either regressing into this file silently.
ALLOWED_LINES = {
    "conf/modules.config": {
        # Line-pinned on purpose (see test_scan_actually_finds_the_consumers): an
        # exemption keyed to a file rather than a line would cover the next raw read
        # added anywhere in it. The cost is that unrelated edits above this line move
        # it -- 718 -> 722 when the reg_qc=2 QC stopped segmenting for itself and
        # SEG_QC_GEOJSON's block shrank; 722 -> 727 when feat/lsa-cell-pairing's
        # rewritten WARP_SEG_QC ext.args comment added 5 net lines above it;
        # 727 -> 736 when the three top-level config helper functions were inlined
        # for Nextflow 26's strict parser; 736 -> 744 when TILED_SOLVE gained the
        # ext.args block carrying its confidence/range gates; 744 -> 748 when that
        # same block's `?:` was replaced by an explicit null test (Elvis treats a
        # legal 0 as unset) and gained the comment explaining why; 748 -> 834 when the
        # in-process BaSiC path became the three-process nf-core BASICPY chain and
        # TILE_FOR_BASIC / BASICPY / APPLY_PROFILES gained withName blocks above it
        # (826 -> 834 once BASICPY's block gained the comment recording that running at
        # upstream defaults is a decision); 834 -> 821 when the dead PREPROCESS withName
        # block was deleted with the in-process BaSiC path; 821 -> 856 when
        # TILE_FOR_BASIC's and APPLY_PROFILES' memory closures stopped being one-line
        # multiples of the input file's size and became multi-line, tile-derived
        # arithmetic (the two processes now stream rather than holding the slide);
        # 856 -> 891 when MERGE_AND_PYRAMID's 200/300 GB tier ladder became the
        # plane-derived closure that followed it into streaming; 891 -> 920 when that
        # closure's comment stopped asserting a 4:1 compression ratio as fact and
        # recorded the measured counterexamples and the retry backstop instead;
        # 920 -> 934 when APPLY_PROFILES' and MERGE_AND_PYRAMID's memory comments
        # each named their `maxworkers=1` dependency on bin/apply_basic_profiles.py
        # and bin/merge_channels_pyramid.py (+5 and +9 net lines respectively) and
        # MERGE_AND_PYRAMID's plane coefficient comment was corrected from 3.11 to
        # 3.25 to agree with its own measured intercept and mechanism sum; 934 -> 936
        # when a review round reworded that same comment (+2 net lines) to stop
        # calling 3.25 a "bound" on every measured point -- C=1 sits 0.18 planes
        # above the C=4/8/16 fit, a finite-C edge effect the +1d adder absorbs;
        # 936 -> 946 when TILE_FOR_BASIC's and APPLY_PROFILES' memory comments lost
        # the "one decoded source plane" file-size term now that CONVERT_IMAGE and
        # SPLIT_CHANNELS write tiled (net +10 lines: the mechanism explanation grew
        # to point at the two tiling tests and record the --prior_outdir / add_cycle
        # legacy-untiled degradation instead of shrinking to nothing), and the C=1
        # gap above was corrected from 0.18 (measured against the 3.25 CODE
        # coefficient plus the per-channel stash term) to 0.21 (measured against the
        # 3.21 FIT the sentence actually names).
        # 946 -> 1034 on THIS branch when the zarr/streaming line was merged into
        # benchmarking: this branch's conf/modules.config carries the
        # benchmarking-only process blocks (+15) AND the restored SEG_QUALITY_EVAL /
        # MERGE_SEG_EVAL blocks (+73), so every line above sits 88 lower. The two
        # offsets COMPOSE exactly -- 946 + 88 = 1034 -- which is the arithmetic check
        # that this re-pin is the merge of both histories rather than a guess at one.
        # 1034 -> 1064 when the registration cost/accuracy presets landed: the three
        # TILED_* closures in conf/modules.config each gained an INLINED copy of the
        # RegPresets.STARE table (+30 lines total), because conf/*.config cannot see
        # lib/*.groovy and the tier params are null-declared. Same composition check as
        # above -- 1034 + 30 = 1064 -- and re-pinned from the file with the grep below,
        # not guessed.
        # 1064 -> 1103 when the ASHLAR benchmark backend landed. TWO offsets compose here:
        # +54 for the three new withName blocks above this line -- ASHLAR_RETILE,
        # ASHLAR_SOLVE (which restates TILED_SOLVE's two publish destinations) and
        # ASHLAR_STITCH (which must restate TILED_STITCH's publishDir, because Nextflow
        # matches a withName selector on the ORIGINAL name and an alias would otherwise
        # inherit it silently) -- and -15 for deleting the byte-identical SECOND copy of
        # the TILED_COARSE block and its comment, which had sat above this line unnoticed
        # (harmless under config merge, last-one-wins). Same composition check as above --
        # 1064 + 54 - 15 = 1103.
        # 1103 -> 1119 when SEGMENT's `clusterOptions` closure (above this line) was fixed
        # to compose --account/--qos instead of replacing them outright: a
        # `withName:` assignment REPLACES process.clusterOptions rather than adding to it,
        # so SEGMENT's GPU-only closure was silently dropping the `slurm` profile's
        # --account/--qos composition. The one-line closure became a 10-line explanatory
        # comment plus a 7-line inlined closure (net +16): 1103 + 16 = 1119. Re-pin, do not
        # widen. (Re-pin from the file, not by guessing:
        # `grep -n "params.expanded_quantification ?" conf/modules.config`.)
        # 1103 -> 1108 when Task 5.1's COARSE front-end selector landed: TILED_COARSE's
        # withName block gained a 4-line comment plus a one-line `ext.args` closure
        # (+5 lines total) above this line. 1103 + 5 = 1108. (That selector, and the
        # parameter it read, were deleted again for v1.0.0 -- see the 1277 -> 1298 entry
        # below. The line names are not quoted here any more because
        # tests/test_no_legacy_frontends.py forbids naming them outside its allow-list.)
        # 1108 -> 1136 when Task 5.4 published the two STARE/VALIS benchmark-scoring
        # artifacts: REGISTER's publishDir gained a third array entry for the VALIS
        # registrar pickle (+11), TILED_REG_TILE's shared block comment was rewritten
        # to explain why its control-point JSON is now published while TILED_COARSE's
        # stays intermediate (+8) and its `publishDir = [ enabled: false ]` became a
        # real publishDir block (+5), and TILED_COARSE gained its own short comment
        # explaining the asymmetry (+4). Four hunks above this line, none of them
        # touching this one: 1108 + 11 + 8 + 5 + 4 = 1136. Re-pin from the file, not by
        # guessing: `grep -n "params.expanded_quantification ?" conf/modules.config`.
        # 1103 -> 1119 on dev when SEGMENT's clusterOptions stopped REPLACING the
        # slurm profile's --account/--qos: +16, the inlined account/qos derivation
        # plus the comment recording why composition is unavailable
        # (task.clusterOptions inside a clusterOptions closure recurses to a
        # StackOverflowError). This is the SAME bug independently fixed on the
        # remediation branch below; the merge keeps one copy of the fix.
        # 1103 -> 1140 on remediation/arch-review-2026-08-24 when failure policy was
        # reduced to a named set (tests/test_error_strategy_policy.py). All +37
        # lines are comment and closure text above this line: -1/+10/+5 rewriting
        # the QC selector's header and its 'ignore' -> 'finish' rationale, +5
        # naming REGISTER's retry-exit1-then-fail policy, +2/+3 correcting the CSE
        # header comment that claimed the closure logs, and +10/+3 replacing the
        # two multi-line log.warn closures with the one-line retry-then-drop
        # policy plus the comment recording why a config closure must not log.
        # Composition check: 1103 + 37 = 1140, and `git diff -U0
        # conf/modules.config` shows every hunk above this line summing to +37.
        # 1140 -> 1156 when MERGE_AND_PYRAMID's memory closure learned that a
        # `path` input is a bare Path for a one-file group: +16, all of it the
        # `instanceof Collection` normalisation and the comment recording the
        # abort it fixes ("No such file or directory: channels").
        # 1156 -> 1181 when SEGMENT's clusterOptions stopped REPLACING the slurm
        # profile's --account/--qos (the remediation branch's own copy of the
        # same fix dev made independently at 1103 -> 1119 above): +25, the
        # inlined account/qos derivation plus the comment recording why
        # composition is unavailable.
        # 1181 -> 1226 when --cleanup_level gained its publishDir gates: +45, being
        # 11 two-line gates above this point (a comment pointer plus one `enabled:`
        # line each) and one 23-line block at the first site recording why the
        # literal is inlined and why it must not be written as a closure.
        # Merging dev (which stopped at 1119, its own copy of the clusterOptions
        # fix) with remediation/arch-review-2026-08-24 (which reached 1226 via a
        # DIFFERENT set of commits, including its own copy of the same
        # clusterOptions fix) collapses the duplicate fix to one copy and lands
        # on whatever `conf/modules.config` actually contains post-merge -- not
        # arithmetic composition of the two divergent histories above, since they
        # are not independent hunks stacked on the same base the way the earlier
        # entries were. Re-pinned directly from the merged file with the grep
        # below, not guessed or computed from the two counts above.
        # 1136 (feat/stare-ultimate) and 1226 (dev) -> 1259 when the two were merged.
        # NOT the sum, and not either input: the branches changed DIFFERENT regions of
        # conf/modules.config above this line, and the merge also collapsed TILED_REG_TILE's
        # and TILED_COARSE's maxForks conflicts into one copy each. Composing the two
        # counts arithmetically would give a number that matches neither file. Re-pinned
        # directly from the merged file, exactly as the dev-side entry above insists:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1259
        #
        # NINE re-pins now sit above this line. That is a standing signal, not a chore:
        # this allowlist keys on POSITION, which every unrelated edit above invalidates,
        # while the thing it means to pin is the line's CONTENT. Keying on the normalised
        # text plus an assertion that it occurs exactly once would be stable under edits
        # above it and would need no re-pin at all.
        # 1259 -> 1270 when the PREFLIGHT_SCALE process (task-2 of the scale-correctness
        # work) gained its own `withName:` block above CONVERT_IMAGE: +11, a publishDir-
        # only block (its resources come from a `label`, not this block -- see the
        # one-owner rule) plus its explanatory comment. Re-pinned directly from the file:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1270
        # 1270 -> 1277 when the pixel_size CRITICAL fix (scale-correctness-and-robustness)
        # added a 7-line comment above SEGMENT's `instantseg` ext.args flags, explaining
        # why `--pixel-size ${meta.pixel_size}` replaced `${params.pixel_size}` there.
        # Re-pinned directly from the file:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1277
        # 1277 -> 1298 when the legacy COARSE front-ends were deleted and TILED_COARSE's
        # memory request stopped being a flat ramp: net +21 above this line, being +5 in
        # the block comment (rewritten for the U-Net cost curve), +21 replacing the
        # one-line `memory = { 8.GB * ... }` with the derived closure and its inlined
        # tier table, and -5 deleting the front-end selector comment and its `ext.args`.
        # 1277 + 5 + 21 - 5 = 1298. Re-pinned directly from the file:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1298
        # 1298 -> 1307 in the same change's review round: +9 lines in TILED_COARSE's block
        # comment, recording why the retry ramp reverted from doubling to linear (the
        # doubling was for a SLIDE-driven peak; the peak is now BOUND-driven, so a retry
        # corrects for host variance rather than searching for an unknown magnitude).
        # 1298 + 9 = 1307. Re-pinned directly from the file:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1307
        # 1307 -> 1251 when the ashlar registration backend was removed for v1.0.0, leaving
        # exactly two production backends (valis, tiled). -56 above this line: the whole
        # ASHLAR section of conf/modules.config -- its banner comment plus the
        # ASHLAR_RETILE, ASHLAR_SOLVE and ASHLAR_STITCH withName blocks and the blank line
        # closing the section. This reverses the +54 half of the 1064 -> 1103 entry above
        # (the section had since grown to 56 lines); the -15 half -- deleting the duplicate
        # TILED_COARSE block -- stands, so this is NOT that entry's inverse and chaining
        # off one would give 1305. Composition check: 1307 - 56 = 1251, and
        # `wc -l conf/modules.config` fell 1394 -> 1338, the same 56. Re-pinned directly
        # from the file, not computed:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1251
        # 1251 -> 1253 in the final whole-branch review: +2 lines in TILED_COARSE's memory
        # closure comment. The old text was a CAUTION telling the operator not to pass
        # `reg_tiled_coarse_max_dim` 0; commit 9073f14 had already made anything below 256
        # a hard launch abort in ParamUtils.validateRegPresets, so the caution described a
        # state the operator can no longer reach and was rewritten to say the floor is
        # enforced. Composition check: 1251 + 2 = 1253, and `wc -l conf/modules.config`
        # rose 1338 -> 1340, the same 2. Re-pinned directly from the file, not computed:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1253
        # 1253 -> 1265 on 2026-08-30 (fix/docs-truthfulness): +12 lines in the "STARE
        # (tiled) per-task memory" header at the TOP of the file. That header asserted
        # "measured peak RSS is 1.3-2.0 GB per task" for EVERY tiled process; the band is
        # the per-tile tasks only and has not covered TILED_COARSE since its coarse anchor
        # became a U-Net (~31.7 GB peak against a 48 GB request at the shipped tier). The
        # replacement paragraph names the exception and the formula it comes from.
        # Composition check: 1253 + 12 = 1265, and `wc -l conf/modules.config` rose
        # 1340 -> 1352, the same 12. Re-pinned directly from the file, not computed:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1265
        # 1265 -> 1256 on 2026-09-02 (release/p02-cleanliness, Task 11): -9 lines
        # compressing four refactor-diary comments above this line (in the
        # TILED_REG_TILE, TILED_COARSE, TILED_STITCH and SPLIT_CHANNELS blocks) down to
        # their trap-guarding content -- the measurements and mechanisms stayed, the
        # "this used to be"/"previously said" framing did not. Composition check:
        # 1265 - 9 = 1256, and `wc -l conf/modules.config` fell 1352 -> 1343, the same 9.
        # Re-pinned directly from the file, not computed:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1256
        # 1256 -> 1260 on 2026-09-03 (release/p08-dapi-overlay, Task 6): +4 lines --
        # GENERATE_REGISTRATION_QC's memory closure gained a comment explaining why its
        # tier now sums THREE inputs (registered + native_image + reference).
        # 1260 -> 1269 at the plan-06 rebase onto dev-with-08 (2026-09-03): +9 more lines
        # ABOVE this line, in the `withName: 'BASICPY'` block -- a `container = '...'`
        # digest pin (ruling R6) plus its 8-line rationale comment, added there rather
        # than in the vendored modules/nf-core/basicpy/main.nf so that file stays
        # byte-for-byte upstream. Composition: 1256 + 4 + 9 = 1269, and
        # `wc -l conf/modules.config` rose 1343 -> 1347 -> 1356. Re-pinned directly
        # from the merged file, not computed:
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1269
        # 1269 -> 1270 (2026-09-06): the cleanup-gate comment on CONVERT_IMAGE's
        # publishDir grew by one line when the gate moved into `saveAs:`.
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1270
        # 1270 -> 1274 (2026-09-09): the four per-process `maxForks` caps moved out
        # of conf/modules.config into nextflow.config's post-profiles concurrency
        # block, each replaced by a two-line pointer comment (+4 net).
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1274
        # 1274 -> 1249 (2026-09-09): the EXTRACT_MASK_SERIES withName block left
        # with the add_cycle feature, which lives on the dev branch only (-25).
        #   grep -n "params.expanded_quantification ?" conf/modules.config  ->  1249
        1249: (
            "ext.args = { params.expanded_quantification ? '--expanded' : "
            "'' } -- conf/*.config closures cannot see lib/*.groovy classes, "
            "so ext.args must read params raw here."
        ),
    },
}


def _is_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith(("//", "*", "/*", "#"))


def _read_sites() -> list[tuple[Path, int, str]]:
    """(file, 1-based line no, line text) for every raw read outside the resolver."""
    sites = []
    for pattern in SCANNED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.name == ROUTER_FILE:
                continue
            for i, line in enumerate(path.read_text().splitlines()):
                if _is_comment(line) or not READ_RE.search(line):
                    continue
                sites.append((path, i + 1, line))
    return sites


def _is_allowed(rel_path: str, lineno: int) -> bool:
    if rel_path in ALLOWED_FILES:
        return True
    return lineno in ALLOWED_LINES.get(rel_path, set())


def test_scan_actually_finds_the_consumers():
    """A scan matching nothing would pass the allowlist check vacuously."""
    sites = _read_sites()
    files = {str(path.relative_to(ROOT)) for path, _no, _line in sites}
    lines = {(str(path.relative_to(ROOT)), no) for path, no, _line in sites}
    assert len(sites) >= 4, (
        f"only {len(sites)} read site(s) found -- globs may be stale"
    )
    for expected in ALLOWED_FILES:
        assert expected in files, (
            f"{expected} no longer reads any of the three compartment params "
            "raw. If it was routed through ParamUtils.compartmentMode() "
            "instead, remove its ALLOWED_FILES entry too -- an allowlist "
            "entry for a file that no longer needs it lets this test go "
            "quiet on the next regression."
        )
    for expected_file, expected_linenos in ALLOWED_LINES.items():
        for expected_lineno in expected_linenos:
            assert (expected_file, expected_lineno) in lines, (
                f"{expected_file}:{expected_lineno} no longer reads a "
                "compartment-quantification param raw. Remove its ALLOWED_LINES "
                "entry too, so a real regression on that exact line cannot hide "
                "behind a stale exemption."
            )


def test_no_file_outside_the_resolver_and_allowlist_reads_it_raw():
    """The seam: every other consumer must hold ParamUtils.compartmentMode()'s
    map, not read params.quantify_compartments / params.expanded_quantification
    / params.embed_masks itself."""
    offenders = [
        f"{path.relative_to(ROOT)}:{no}: {line.strip()}"
        for path, no, line in _read_sites()
        if not _is_allowed(str(path.relative_to(ROOT)), no)
    ]
    assert not offenders, (
        f"{len(offenders)} site(s) read a compartment-quantification param raw "
        "outside lib/ParamUtils.groovy (the resolver) and the documented "
        "allowlist. Resolve it once via ParamUtils.compartmentMode(params) and "
        "thread the result down as an argument instead -- the same seam "
        "--registration_method has (subworkflows/local/registration.nf) -- "
        "rather than re-reading the raw param here.\n" + "\n".join(offenders)
    )
