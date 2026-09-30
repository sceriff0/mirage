"""Guards for `nextflow -resume`: every fan-in must order by data, never by arrival.

WHY THIS FILE EXISTS. Five identical stub runs of this pipeline cached 22, 8, 22, 22
and 22 of 28 tasks. Nothing changed between them. The 8 came from
`registration.nf`'s `.groupTuple()` handing REGISTER its slides in a different order,
which changed the task hash (Nextflow hashes list inputs POSITIONALLY) and cascaded a
re-run of everything downstream. On a real WSI run that is days of recomputation, and
it is invisible -- the run is green either way.

Two of these checks guard something worse than a cache miss:

  * `groupTuple(sort:)` orders each grouped list INDEPENDENTLY, so metas and files are
    sorted by their own natural orders and silently re-paired. Demonstrated:
        in:  ['k','zeta','a.txt'], ['k','alpha','z.txt']   # zeta<->a.txt, alpha<->z.txt
        out: [k, [alpha, zeta], [a.txt, z.txt]]            # alpha now <-> a.txt
    It is the fix a reader reaches for first, and it is wrong. Pair first (transpose),
    then sort the pairs.

  * A bare `params` inside a process `script:` block puts the WHOLE params map into
    that task's hash, so ANY parameter change anywhere re-runs it. Measured: changing
    only `--pyramid_resolutions` (a postprocessing knob) re-ran REGISTER and SEGMENT
    and cascaded 20 of 28 tasks.

These are STATIC checks: they assert the ordering mechanism is present at each site.
They are cheap, run in CI's blocking python-tests gate, and cannot pass by luck --
unlike a run-twice cache measurement, where an intermittent regression (the REGISTER
miss showed up in roughly one run in four) can go unnoticed. tests/resume_check.sh is
the behavioural counterpart; see its header for why it is not in the gate.
"""

import re

from tests.nfmodel import REPO_ROOT as ROOT
from tests.nfmodel import nf_files as _nf_files
from tests.nfmodel import strip_comments as _strip_comments_only
from tests.nfmodel import strip_comments_and_strings as _strip_comments

# --------------------------------------------------------------------------- #
# 1. The pairing trap
# --------------------------------------------------------------------------- #


def test_group_tuple_sort_option_is_never_used():
    """groupTuple(sort:) re-pairs meta with the wrong file. Transpose-then-sort instead."""
    offenders = []
    for f in _nf_files():
        body = _strip_comments(f.read_text())
        for i, line in enumerate(body.splitlines(), 1):
            if re.search(r"groupTuple\s*\([^)]*\bsort\s*:", line):
                offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "groupTuple(sort:) sorts each grouped list INDEPENDENTLY, so a [key, metas, files] "
        "group has its metas and files ordered separately and silently re-paired -- meta[0] "
        "ends up describing a different file. Pair first with transpose(), then sort the "
        "pairs (see subworkflows/local/add_cycle.nf's tiff grouping for the idiom).\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# 2. Every fan-in that reaches a process input orders by data
# --------------------------------------------------------------------------- #

# site -> (file, a regex the file MUST match, why)
ORDERED_FAN_INS = {
    "registration: REGISTER's slide list": (
        "subworkflows/local/registration.nf",
        r"\[metas,\s*files\]\.transpose\(\)\.toSorted",
        "REGISTER receives preproc_files/all_metas positionally; arrival order changed its hash.",
    ),
    "quantify_markers: MERGE_AND_PYRAMID's tiff list": (
        "subworkflows/local/quantify_markers.nf",
        r"tiffs_unordered\.toSorted",
        "MERGE_AND_PYRAMID stages the tiff list positionally.",
    ),
    "quantify_markers: MERGE_QUANT_CSVS' csv list and metas[0]": (
        "subworkflows/local/quantify_markers.nf",
        r"\[metas_unordered,\s*csvs_unordered\]\.transpose\(\)\s*\n?\s*\.toSorted",
        "metas[0] fed channels/is_reference into MERGE_QUANT_CSVS and both exports.",
    ),
    "tiled_adapter: TILED_SOLVE's control list and metas[0]": (
        "subworkflows/local/adapters/tiled_adapter.nf",
        r"\[metas,\s*controls\]\.transpose\(\)\.toSorted",
        "Same metas[0] shape as quantify_markers, on the STARE backend.",
    ),
}


def test_every_fan_in_orders_by_data():
    missing = []
    for name, (relpath, pattern, why) in ORDERED_FAN_INS.items():
        body = _strip_comments((ROOT / relpath).read_text())
        if not re.search(pattern, body):
            missing.append(f"{name} ({relpath}): expected /{pattern}/ -- {why}")
    assert not missing, (
        "A fan-in lost its canonical ordering. groupTuple emits in ARRIVAL order and "
        "Nextflow hashes list inputs POSITIONALLY, so this makes -resume miss and cascade.\n  "
        + "\n  ".join(missing)
    )


# --------------------------------------------------------------------------- #
# 3. collect()/collectFile() feeding a process must be sorted
# --------------------------------------------------------------------------- #


def test_final_qc_collects_are_sorted():
    body = _strip_comments_only((ROOT / "subworkflows/local/final_qc.nf").read_text())
    # Real calls are 3-arg -- artifactsOf(ch_artifacts, 'kind', consumed_kinds) -- since
    # f892012 added consumed_kinds. A 2-arg-only pattern here matches zero real call
    # sites regardless of whether sort: true is present, so it can never fail: verified
    # by planting a bare .collect() on a real 3-arg call and confirming the 2-arg
    # version stayed green while this one catches it.
    unsorted = re.findall(
        r"artifactsOf\(ch_artifacts, '([a-z_]+)'(?:,\s*\w+)?\)\.collect\(\)", body
    )
    assert not unsorted, (
        f"GENERATE_QC_REPORT slot(s) {sorted(unsorted)} use a bare .collect(). Its inputs are "
        "path collections hashed POSITIONALLY, and collect() emits in arrival order, so the "
        "report re-ran on every identical rerun. Use .collect(sort: true)."
    )
    assert re.search(
        r"collectFile\(name: 'collated_versions\.yml', sort: true\)", body
    ), (
        "collated_versions.yml lost `sort: true`. Without it the collected yaml's LINE ORDER "
        "varies by task completion order, so its content -- not just its mtime -- differs "
        "between identical runs."
    )


def test_postprocess_reg_qc_collects_are_sorted():
    body = _strip_comments((ROOT / "subworkflows/local/postprocess.nf").read_text())
    bare = re.findall(r"ch_reg_(?:qc|residuals)[^\n]*\.collect\(\)", body)
    assert not bare, (
        f"EXPORT_SPATIALDATA's registration-QC input(s) use a bare .collect(): {bare}. "
        "Use .collect(sort: true) -- these become `path` inputs hashed positionally."
    )


def test_input_check_preflight_scale_collect_is_sorted():
    body = _strip_comments((ROOT / "subworkflows/local/input_check.nf").read_text())
    assert re.search(r"PREFLIGHT_SCALE\([^)]*\.collect\(sort:\s*true\)\)", body), (
        "PREFLIGHT_SCALE's image list uses a bare .collect(). It becomes a `path` input "
        "hashed positionally, so an identical rerun can miss and cascade. Use "
        ".collect(sort: true)."
    )


# --------------------------------------------------------------------------- #
# 4. No process script may hash the whole params map
# --------------------------------------------------------------------------- #


def test_no_process_script_references_the_whole_params_map():
    """`ParamUtils.foo(params)` in a script: block binds the task to EVERY parameter."""
    offenders = []
    for f in sorted((ROOT / "modules").rglob("*.nf")):
        body = _strip_comments(f.read_text())
        for i, line in enumerate(body.splitlines(), 1):
            # a bare `params` passed as a value: `f(params)`, `f(params,`, `key: params]`
            if re.search(r"[(,:]\s*params\s*[,)\]]", line):
                offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "A process passes the WHOLE params map as a value. Nextflow hashes the free variables "
        "a `script:` block references, so this binds the task's cache key to every parameter in "
        "the pipeline and any unrelated change re-runs it and everything downstream. Measured: "
        "`--pyramid_resolutions 6` re-ran REGISTER and SEGMENT and cascaded 20 of 28 tasks. "
        "Pass the specific values instead -- see ParamUtils.regQcLevelOf / "
        "SegBackends.CTX_PARAM_KEYS.\n  " + "\n  ".join(offenders)
    )
