"""Make the claims in `docs/figures/*.html` checkable against the pipeline.

Those figures are the supplementary methods explainers, which a reviewer reads
instead of the source; a static HTML page fails no test unless something parses it.

Four claim classes are checked, chosen because a test can settle them:

1. **Parameter names.** Every `--<name>` and `params.<name>` a figure writes
   must be declared in `nextflow.config`'s `params {}` block.
2. **Pipeline names.** Two scans, unioned. A ``UPPER_SNAKE_CASE`` token
   anywhere in the figure, and *any* all-caps token written as code
   (``<code>``, ``.nm``, ``.name``) -- the second is what puts single-word
   names like ``REGISTER`` and ``SEGMENT`` under the check. Each must resolve
   to a process, a named workflow/subworkflow, or an aliased include.
3. **Parameter rows.** Every ``<td class="k">`` key inside a ``<table
   class="pr">`` that reads as a parameter key must name a declared parameter.
4. **Parameter defaults.** Every such row must state that parameter's actual
   `nextflow.config` default.

What this does **not** cover, and cannot:

* **Prose.** "VALIS matches with SuperGlue", "the micro pass runs twice" -- a
  claim about behaviour, not about a name or a number. Those are what the
  allow-listed figure below is stale on, and no parser settles them.
* **Numbers that are not parameter defaults.** Measured memory figures, file
  sizes, timings, tier tables written as free text rather than as a `k`/`v`
  row. They are checked by nothing here.
* **Whether a figure is COMPLETE.** A figure that omits a whole stage passes.
* **Diagram topology.** The arrows are SVG; that a name exists says nothing
  about whether it is drawn in the right place.
* **A single-word pipeline name written as bare prose.** `REGISTER` inside
  `<code>` is checked; the same word in a running sentence is not, because a
  figure legitimately writes SEGMENT, REGISTER, INPUT and STEP as English.
  Marking a name as code is the claim this file can hold you to.
* **The VALUE half of a compound parameter row.** A key like `seg_pmin /
  seg_pmax` carries two defaults in one cell (`1.0 / 99.8`). Both NAMES are
  checked for existence; neither value is compared.

The floors in each test stop the checks becoming vacuous if the markup is
restyled and the extraction silently matches nothing.
"""

from __future__ import annotations

import html
import importlib.util
import re
import sys
from pathlib import Path

from tests.nfmodel import REPO_ROOT, include_aliases, processes, workflows

FIGURES_DIR = REPO_ROOT / "docs" / "figures"

# --------------------------------------------------------------------------
# The shrink-only debt allowlist.
#
# An entry is a debt, not a permission: `test_the_allowlist_only_shrinks` fails
# if an allow-listed figure stops failing. Fix a failing figure; do not add it here.
# --------------------------------------------------------------------------
STALE_FIGURES = {}

# `--<token>` forms that are NOT pipeline parameters. Each is another tool's
# flag, quoted in a figure because the pipeline renders it. Kept honest by
# `test_non_param_flags_are_really_not_params`: an entry must not name a real
# parameter (so it can never shadow one) and must still appear in a figure
# (so it cannot rot into a blanket).
NON_PARAM_FLAGS = {
    "gres": "SLURM's own flag, rendered by conf/modules.config as --gres=gpu:$gpu_type",
    "gpus": "Docker's flag, added to the run command when seg_gpu is true",
    "nv": "Singularity's flag, the --gpus equivalent",
    "method": "bin/warp_seg_qc.py's CLI flag, not a pipeline parameter",
    "tolerance": "bin/seg_qc_geojson.py's CLI flag, not a pipeline parameter",
    "pairing": "bin/warp_seg_qc.py's CLI flag; the pipeline parameter is seg_qc_pairing",
    # Hyphenated flags (e.g. `--match-radius-factor`) never match _FLAG, so need no entry.
}

# ``UPPER_SNAKE_CASE`` tokens that name something real but are not Nextflow
# process or workflow names. Kept honest the same way, by
# `test_non_pipeline_names_are_really_not_pipeline_names`.
NON_PIPELINE_NAMES = {
    "DEEPCELL_ACCESS_TOKEN": "an environment variable read by the DeepCell backend",
    "KNOWN_ARTIFACT_KINDS": "a lib/ParamUtils.groovy constant",
    "STEP_ORDER": "a lib/ParamUtils.groovy constant",
    "STEPS": "a lib/ParamUtils.groovy constant, written as `ParamUtils.STEPS`",
    "UNIVERSAL_QC_KINDS": "a lib/ParamUtils.groovy constant",
    "DEFAULT_BAND_BYTES": "a bin/utils constant",
    "RUSAGE_SELF": "Python's resource.RUSAGE_SELF",
    # Reached by the code-scoped scan below, which is not restricted to
    # underscore-bearing tokens and so sees ordinary all-caps literals.
    "DAPI": "a channel/marker name, not a pipeline name",
    "CD3": "a channel/marker name, not a pipeline name",
    "CELLTOX": "a channel/marker name, not a pipeline name",
    "CYX": 'a TIFF axis order, written as `axes="CYX"`',
    "PREPROCESS": "the published subdirectory `qc/preprocess`, not a process",
}

# `<td class="k">` keys inside a `<table class="pr">` that read as a parameter
# key but name no parameter. Same honesty rule as the two lists above: an entry
# must NOT be a declared parameter, and must still appear in a checked figure.
NOT_A_PARAM_ROW = {
    "max_dim": "a Python keyword argument of the lazy readers, not a parameter",
    "factor": "a decimation factor computed at runtime, not a parameter",
    "is_reference": (
        "a samplesheet COLUMN, not a parameter -- registration-schematic.html's "
        "dispatch table lists it beside the params that govern the same decision, "
        "and says so in its own description cell"
    ),
}

# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

# CSS custom properties are spelled `--name` too. Every one of them lives in a
# <style> block, a style="" attribute or a var(--name) reference, so removing
# those three is what separates a design token from a claim about the pipeline.
_STYLE_BLOCK = re.compile(r"<style\b.*?</style>", re.S | re.I)
_STYLE_ATTR = re.compile(r"""\sstyle\s*=\s*(["']).*?\1""", re.S | re.I)
_VAR_REF = re.compile(r"var\(\s*--[A-Za-z0-9_-]+\s*\)")
# HTML comments cannot nest, so a non-greedy DOTALL match is the whole comment.
# Positive checks read `_prose()` so commented-out markup cannot satisfy them; the
# negative guard in tests/test_figures_have_no_retired_names.py reads raw text.
_COMMENT = re.compile(r"<!--.*?-->", re.S)

# A Nextflow parameter is snake_case with no hyphen, so Python CLI flags such as
# `--jvm-heap-gb` are excluded structurally rather than by name.
_FLAG = re.compile(r"--([a-z][a-z0-9_]*)(?![A-Za-z0-9_-])")
_PARAMS_DOT = re.compile(r"params\.([a-z][a-z0-9_]*)")
# Two complementary name scans; the union is what gets resolved.
#
# `_UPPER` runs over the whole figure, so it catches a name in bare prose. It
# requires an underscore, or it would match every all-caps English word.
#
# `_CODE_SPAN`: anything marked up as code is a claim that the identifier exists,
# so the underscore requirement is dropped there. That covers single-word process
# names such as REGISTER and SEGMENT.
_UPPER = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_CODE_SPAN = re.compile(
    r"<code[^>]*>(.*?)</code>"
    r"|<(?:div|span)[^>]*class=\"(?:nm|name)\"[^>]*>(.*?)</(?:div|span)>",
    re.S,
)
_CODE_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]{2,}(?![A-Za-z0-9_])")
# The parameter tables. Every other k/v table in these figures (`qa`, `rc`,
# `own`, `ss`, `bk`) pairs prose with prose.
_PR_TABLE = re.compile(r'<table class="pr">(.*?)</table>', re.S)

_TAG = re.compile(r"<[^>]+>")
# Key and value cells of one parameter row. Neither capture may span a `</td>`,
# so a `k` cell without a `v` cell cannot swallow the rows after it.
_KV_ROW = re.compile(
    r'<td class="k">((?:(?!</td>).)*)</td>\s*<td class="v[^"]*">((?:(?!</td>).)*)</td>',
    re.S,
)


def _figures():
    found = sorted(FIGURES_DIR.glob("*.html"))
    assert found, f"no figures found under {FIGURES_DIR}"
    return found


def _live_figures():
    return [f for f in _figures() if f.name not in STALE_FIGURES]


def _prose(path: Path) -> str:
    """Figure text with CSS custom properties AND HTML comments removed.

    Every check that asks "does the figure claim X" is a positive rule, so a row
    or name inside `<!-- ... -->` must not count as a claim.
    """
    text = path.read_text()
    return _COMMENT.sub(
        " ", _VAR_REF.sub(" ", _STYLE_ATTR.sub(" ", _STYLE_BLOCK.sub(" ", text)))
    )


def _cell_text(fragment: str) -> str:
    return html.unescape(_TAG.sub("", fragment)).replace("\xa0", " ").strip()


def test_a_param_row_inside_an_html_comment_is_not_counted(tmp_path):
    """`_prose()` must blank HTML comments, or a commented-out row/name would
    satisfy a POSITIVE check without the figure ever making that claim.
    """
    fixture = tmp_path / "commented.html"
    fixture.write_text(
        "<html><body>\n"
        "<!--\n"
        '<table class="pr"><tr>'
        '<td class="k">reg_qc</td><td class="v">2</td>'
        "</tr></table>\n"
        "REGISTER_PATIENT\n"
        "-->\n"
        "</body></html>\n"
    )
    assert list(_param_rows(fixture)) == []
    assert "REGISTER_PATIENT" not in _names_used(fixture)


def _config_defaults():
    """`nextflow.config`'s params {} block, via the existing reader.

    `tests/check_param_consistency.py` owns this parse. It is a script, not a
    module on the path, so it is loaded the way
    `tests/test_param_consistency_rule.py` loads it.
    """
    spec = importlib.util.spec_from_file_location(
        "check_param_consistency", REPO_ROOT / "tests" / "check_param_consistency.py"
    )
    module = importlib.util.module_from_spec(spec)
    # It imports `_code_view` flat, so put tests/ on sys.path explicitly rather than
    # relying on pytest's rootdir insertion.
    if str(REPO_ROOT / "tests") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
    spec.loader.exec_module(module)
    defaults = module.extract_config_defaults(
        (REPO_ROOT / "nextflow.config").read_text()
    )
    assert len(defaults) > 50, (
        f"only {len(defaults)} params parsed out of nextflow.config -- the "
        "reader has stopped working and every check below would pass vacuously"
    )
    return defaults, module.Unparsed


def _known_names():
    """Every UPPER_SNAKE_CASE name the pipeline actually defines."""
    return set(processes()) | set(workflows()) | set(include_aliases())


def _rendered(value) -> set:
    """The spellings a figure may legitimately use for a config default."""
    if value is None:
        return {"null", "none"}
    if isinstance(value, bool):
        return {"true" if value else "false"}
    if isinstance(value, list):
        inner = ", ".join(str(v) for v in value)
        return {f"[{inner}]", inner}
    if isinstance(value, float):
        out = {str(value)}
        if value.is_integer():
            out.add(str(int(value)))
        return out
    text = str(value)
    return {text, f"'{text}'", f'"{text}"'}


# --------------------------------------------------------------------------
# 1. parameter names
# --------------------------------------------------------------------------
def _flag_problems(path, defaults):
    prose = _prose(path)
    named = {m.group(1) for m in _FLAG.finditer(prose)}
    named |= {m.group(1) for m in _PARAMS_DOT.finditer(prose)}
    return [
        f"{path.name}: names `--{name}`/`params.{name}`, which nextflow.config's "
        "params {} block does not declare"
        for name in sorted(named)
        if name not in defaults and name not in NON_PARAM_FLAGS
    ]


def test_every_parameter_a_figure_names_exists():
    defaults, _ = _config_defaults()
    problems, seen = [], 0
    for path in _live_figures():
        problems += _flag_problems(path, defaults)
        seen += len({m.group(1) for m in _FLAG.finditer(_prose(path))} & set(defaults))
    assert seen >= 8, (
        f"only {seen} parameter mentions found across the figures -- the "
        "extractor has stopped matching and this test is checking nothing"
    )
    assert not problems, "\n".join(problems)


def test_non_param_flags_are_really_not_params():
    """Each NON_PARAM_FLAGS entry must name something that is NOT a parameter
    (an exemption must never be able to hide a real one) and must still appear
    in a figure this file checks. An entry that stops matching is exempting
    nothing and is deleted -- the shrink-only rule, applied to this list."""
    defaults, _ = _config_defaults()
    prose = " ".join(_prose(p) for p in _live_figures())
    used = {m.group(1) for m in _FLAG.finditer(prose)}
    problems = []
    for name in NON_PARAM_FLAGS:
        if name in defaults:
            problems.append(
                f"--{name} IS a declared parameter now -- remove it from "
                "NON_PARAM_FLAGS, it is exempting a real one"
            )
        if name not in used:
            problems.append(
                f"--{name} no longer appears in any checked figure -- remove it "
                "from NON_PARAM_FLAGS"
            )
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------
# 2. pipeline names
# --------------------------------------------------------------------------
def _names_used(path):
    """Every token in this figure that is a claim about a pipeline name.

    Reads `_prose()`, not `path.read_text()`, so a name sitting only inside a
    `<!-- ... -->` section-label comment (e.g. `<!-- e — SEG_QC -->`) does not
    count as the figure claiming that name exists.
    """
    text = _prose(path)
    named = {m.group(0) for m in _UPPER.finditer(text)}
    for span in _CODE_SPAN.finditer(text):
        fragment = _TAG.sub("", span.group(1) or span.group(2) or "")
        named |= {m.group(0) for m in _CODE_TOKEN.finditer(fragment)}
    return named


def _name_problems(path, known):
    return [
        f"{path.name}: names `{name}`, which is not a process, a workflow or an "
        "aliased include anywhere in the pipeline"
        for name in sorted(_names_used(path))
        if name not in known and name not in NON_PIPELINE_NAMES
    ]


def test_every_pipeline_name_a_figure_uses_exists():
    """A figure naming a process that was renamed or archived is the drift this
    catches. Resolution is against processes + workflows + include aliases,
    because a subworkflow and an aliased include are pipeline names too."""
    known = _known_names()
    problems, seen = [], set()
    for path in _live_figures():
        problems += _name_problems(path, known)
        seen |= _names_used(path) & known
    assert len(seen) >= 20, (
        f"only {len(seen)} known pipeline names found across the figures -- the "
        "extractor has stopped matching"
    )
    # The single-word case, pinned by name: both are real processes named
    # repeatedly in the figures, and only the code-scoped scan sees them.
    missing = {"REGISTER", "SEGMENT"} - seen
    assert not missing, (
        f"{sorted(missing)} no longer reach the resolver. Either the process was "
        "renamed (fix the figures) or the code-scoped scan stopped matching (fix "
        "this file) -- single-word names must not fall out of coverage again."
    )
    assert not problems, "\n".join(problems)


def test_non_pipeline_names_are_really_not_pipeline_names():
    """The NON_PIPELINE_NAMES counterpart of the flag check above."""
    known = _known_names()
    used = set()
    for path in _live_figures():
        used |= _names_used(path)
    problems = []
    for name in NON_PIPELINE_NAMES:
        if name in known:
            problems.append(
                f"{name} IS a pipeline name now -- remove it from "
                "NON_PIPELINE_NAMES, it is exempting a real one"
            )
        if name not in used:
            problems.append(
                f"{name} no longer appears in any checked figure -- remove it "
                "from NON_PIPELINE_NAMES"
            )
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------
# 3. parameter defaults
# --------------------------------------------------------------------------
def _param_rows(path):
    """`(key, value)` for every k/v row inside a `<table class="pr">`.

    Scoped to that one table class because it is the only one these figures use
    for parameters; `qa`, `rc`, `own`, `ss` and `bk` pair prose with prose
    (`step 1 · features`, `eager whole-array write`). Unscoped, every prose row
    would have to be excused by name.

    Reads `_prose()`, not `path.read_text()`, so a `<table class="pr">` row
    sitting inside an HTML comment does not count as a claim the figure makes.
    """
    for table in _PR_TABLE.finditer(_prose(path)):
        for m in _KV_ROW.finditer(table.group(1)):
            yield _cell_text(m.group(1)), _cell_text(m.group(2))


def _row_keys(key):
    """The parameter names a row key claims. `a / b` claims two."""
    return [
        part.strip()
        for part in key.split("/")
        if re.fullmatch(r"[a-z][a-z0-9_]*", part.strip())
    ]


def _row_key_problems(path, defaults):
    """A parameter-table key that reads as a param key but names none.

    Such a row escapes the name check above (a table key carries no
    `--`/`params.` prefix), so it is reported here.
    """
    problems = []
    for key, _value in _param_rows(path):
        for name in _row_keys(key):
            if name not in defaults and name not in NOT_A_PARAM_ROW:
                problems.append(
                    f"{path.name}: parameter table row `{key}` names `{name}`, "
                    "which nextflow.config's params {} block does not declare"
                )
    return problems


def _default_rows(path, defaults):
    """`(param, figure_value)` for every parameter row carrying ONE default.

    A compound key (`seg_pmin / seg_pmax`, value `1.0 / 99.8`) has both of its
    NAMES checked by `_row_key_problems`; splitting its VALUE is bespoke per
    row and is not attempted -- see the module docstring.
    """
    for key, value in _param_rows(path):
        if "/" in key:
            continue  # compound: the value cell carries two defaults, not one
        names = _row_keys(key)
        if len(names) == 1 and names[0] in defaults:
            yield names[0], value


def _value_problems(path, defaults, unparsed_type):
    problems = []
    for key, shown in _default_rows(path, defaults):
        expected = defaults[key]
        if isinstance(expected, unparsed_type):
            continue  # a Groovy expression the config reader cannot evaluate
        # `…`/`...` is the figures' explicit "truncated for width" marker.
        if shown.endswith(("…", "...")):
            stem = shown.rstrip(".…").strip()
            if stem and any(s.startswith(stem) for s in _rendered(expected)):
                continue
        if shown not in _rendered(expected):
            problems.append(
                f"{path.name}: states `{key} = {shown}` but nextflow.config "
                f"declares {expected!r}"
            )
    return problems


def test_every_parameter_row_key_is_a_real_parameter():
    """A misspelled key in a parameter table must FAIL, not vanish.

    E.g. `<td class="k">reg_qcc</td><td class="v">7</td>`: the key carries no
    `--` prefix for the name check, and an unknown key has no default to compare.
    """
    defaults, _ = _config_defaults()
    problems, keys = [], 0
    for path in _live_figures():
        problems += _row_key_problems(path, defaults)
        keys += sum(len(_row_keys(k)) for k, _ in _param_rows(path))
    assert keys >= 60, (
        f"only {keys} parameter-table keys extracted -- the table markup has "
        "changed and this test would pass on an empty loop"
    )
    assert not problems, "\n".join(problems)


def test_not_a_param_rows_are_really_not_params():
    """The NOT_A_PARAM_ROW counterpart of the two exemption checks above."""
    defaults, _ = _config_defaults()
    used = set()
    for path in _live_figures():
        for key, _value in _param_rows(path):
            used |= set(_row_keys(key))
    problems = []
    for name in NOT_A_PARAM_ROW:
        if name in defaults:
            problems.append(
                f"{name} IS a declared parameter now -- remove it from "
                "NOT_A_PARAM_ROW, it is exempting a real one"
            )
        if name not in used:
            problems.append(
                f"{name} no longer appears in a checked parameter table -- "
                "remove it from NOT_A_PARAM_ROW"
            )
    assert not problems, "\n".join(problems)


def test_every_default_a_figure_states_matches_nextflow_config():
    """The drift with teeth: a figure advertising a default that was changed.

    `nextflow.config` is the single source of truth for defaults; a figure is a
    second declaration that cannot be executed.
    """
    defaults, unparsed = _config_defaults()
    problems, rows = [], 0
    for path in _live_figures():
        rows += len(list(_default_rows(path, defaults)))
        problems += _value_problems(path, defaults, unparsed)
    assert rows >= 60, (
        f"only {rows} parameter-default rows extracted from the figures -- the "
        "table markup has changed and this test would pass on an empty loop"
    )
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------
# 4. registration backend names
# --------------------------------------------------------------------------
# The name check says nothing about the lower-case method values the figures
# write in prose and `registration_method` rows; lib/RegBackends.groovy owns
# that list, and a figure naming a method it lacks fails here.
#
# Two exclusions, matching tests/test_figures_have_no_retired_names.py's
# carve-out: "ASHLAR's" (a possessive reference to the external tool) and
# "Elastix bindings" (VALIS's optional refiner, which the image does not carry).
# A possessive is still flagged when a shipping verb sits within 40 chars before it.
_METHOD_WORD = re.compile(
    r"(?<![A-Za-z0-9_])(valis|tiled|ashlar|stare_ml|elastix)(?![A-Za-z0-9_])"
    r"(?!'s)(?! bindings)",
    re.I,
)
_SHIP_VERB = re.compile(r"\b(?:ships|supports|adds|selectable)\b", re.I)
_POSSESSIVE_METHOD = re.compile(
    r"(?<![A-Za-z0-9_])(valis|tiled|ashlar|stare_ml|elastix)(?![A-Za-z0-9_])'s", re.I
)


def test_figures_name_exactly_the_shipped_registration_backends():
    from tests.test_reg_backends import reg_backend_methods

    shipped = set(reg_backend_methods())
    assert shipped, "RegBackends.methods() parsed as empty -- the reader is broken"

    named, per_file = set(), {}
    for path in _live_figures():
        prose = _prose(path)
        found = {m.group(1).lower() for m in _METHOD_WORD.finditer(prose)}
        for m in _POSSESSIVE_METHOD.finditer(prose):
            if _SHIP_VERB.search(prose[max(0, m.start() - 40) : m.start()]):
                found.add(m.group(1).lower())
        per_file[path.name] = found
        named |= found

    assert named, (
        "no registration-method word found in any figure -- the extractor has "
        "stopped matching and this test is checking nothing"
    )
    extra = named - shipped
    assert not extra, (
        f"figure(s) name registration backend(s) the pipeline does not ship: "
        f"{sorted(extra)} (RegBackends.methods() == {sorted(shipped)}). "
        f"Per file: { {k: sorted(v) for k, v in per_file.items() if v & extra} }"
    )
    missing = shipped - named
    assert not missing, (
        f"the figure set names no backend {sorted(missing)}, which the pipeline "
        f"ships. A supplementary set that omits a whole backend is a claim that "
        f"it does not exist."
    )


def test_a_shipping_claim_is_not_excused_by_a_possessive_tail():
    """Drill for the residual noted above _METHOD_WORD: a possessive form
    dressed up as a shipping claim ("mirage ships ashlar's tile solver")
    must still be caught, even though a bare possessive ("mirrors ASHLAR's")
    must not be."""
    adversarial = "mirage ships ashlar's tile solver end to end."
    m = next(_POSSESSIVE_METHOD.finditer(adversarial))
    assert _SHIP_VERB.search(adversarial[max(0, m.start() - 40) : m.start()])

    descriptive = "the kernel mirrors ASHLAR's phase-correlation approach."
    m = next(_POSSESSIVE_METHOD.finditer(descriptive))
    assert not _SHIP_VERB.search(descriptive[max(0, m.start() - 40) : m.start()])


# --------------------------------------------------------------------------
# the allowlist meta-test
# --------------------------------------------------------------------------
def test_the_allowlist_only_shrinks():
    """A STALE_FIGURES entry must name a figure that exists and that still
    fails at least one check above.

    An entry that has been fixed but not removed would let the next regression
    in that figure land silently.
    """
    defaults, unparsed = _config_defaults()
    known = _known_names()
    stale = []
    for name in STALE_FIGURES:
        path = FIGURES_DIR / name
        if not path.is_file():
            stale.append(f"{name}: no longer exists -- remove it from STALE_FIGURES")
            continue
        problems = (
            _flag_problems(path, defaults)
            + _name_problems(path, known)
            + _row_key_problems(path, defaults)
            + _value_problems(path, defaults, unparsed)
        )
        if not problems:
            stale.append(
                f"{name}: now passes every check -- remove it from STALE_FIGURES "
                "so the next drift in it is caught"
            )
    assert not stale, "\n".join(stale)


def test_the_allowlist_is_not_a_blanket():
    """At least one figure must be checked. An allowlist that grew to cover
    every figure would leave all five tests above iterating over nothing."""
    assert len(_live_figures()) >= len(_figures()) - 1
