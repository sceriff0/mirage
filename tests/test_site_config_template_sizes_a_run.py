"""`conf/site.config.template` must supply the parameters that have no default.

`nextflow.config` declares `max_cpus` and `max_memory` as `null` on purpose (see
the comment above `max_cpus = null`: a default there is a lie about the machine),
and `nextflow_schema.json`'s `resource_limits` group lists both under `required`.
A run that sets neither is refused before any process is submitted:

    ERROR ~ Validation of pipeline parameters failed!
    * Missing required parameter(s): max_cpus, max_memory

Every documented command satisfies them with `-c site.config`, copied from this
template, so a template without them makes every documented command refuse to launch.

The rule is derived, not a hardcoded pair: every parameter that the schema marks
required and that `nextflow.config` declares `null` must appear in the template,
apart from the per-run values a site template cannot know.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "conf" / "site.config.template"

# Required-and-defaultless parameters a SITE template cannot supply, because they
# are per-run. Each is asserted below to still be required-and-defaultless, so a
# stale entry cannot quietly cover a real omission.
PER_RUN = {
    "input": "the samplesheet path is chosen per run, not per site",
}


def _schema_required():
    """Every parameter name listed in any `required:` array in the schema."""
    schema = json.loads((REPO / "nextflow_schema.json").read_text())
    required = set()

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("required"), list):
                required.update(n for n in node["required"] if isinstance(n, str))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert required, (
        "schema walk found no required params -- the guard would pass vacuously"
    )
    return required


def _config_defaults():
    """`nextflow.config`'s params {} block, via the reader that already owns it.

    `tests/check_param_consistency.py` is a script rather than a module on the
    path, so it is loaded the way tests/test_figures_match_the_pipeline.py loads
    it -- rather than growing a second Groovy-literal parser here.
    """
    spec = importlib.util.spec_from_file_location(
        "check_param_consistency", REPO / "tests" / "check_param_consistency.py"
    )
    module = importlib.util.module_from_spec(spec)
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    spec.loader.exec_module(module)
    defaults = module.extract_config_defaults((REPO / "nextflow.config").read_text())
    assert len(defaults) > 50, (
        f"only {len(defaults)} params parsed out of nextflow.config -- the reader has "
        "stopped working and every check below would pass vacuously"
    )
    return defaults


def _must_be_in_the_template():
    defaults = _config_defaults()
    required = _schema_required()
    return {
        name
        for name in required
        if name in defaults and defaults[name] is None and name not in PER_RUN
    }


def _template_assignments():
    """Parameter names assigned inside the template's `params { ... }` block."""
    text = TEMPLATE.read_text()
    start = text.find("params {")
    assert start != -1, "conf/site.config.template has no params { } block at all"
    depth = 0
    end = None
    for i in range(text.find("{", start), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, "conf/site.config.template's params { } block is unbalanced"
    body = text[start:end]
    body = re.sub(r"//[^\n]*", "", body)
    return set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", body, re.M))


def test_the_template_supplies_every_required_parameter_with_no_default():
    expected = _must_be_in_the_template()
    assert expected, (
        "no required-and-defaultless parameter was found -- either the schema's "
        "`required` arrays or nextflow.config's null declarations stopped being "
        "read, and this guard would pass vacuously"
    )
    missing = sorted(expected - _template_assignments())
    assert not missing, (
        "conf/site.config.template does not set "
        + ", ".join(missing)
        + ". These have no default and are required by nextflow_schema.json, so "
        "every documented `nextflow run ... -c site.config` would be refused at "
        "launch with 'Missing required parameter(s)'."
    )


def test_the_per_run_exemptions_are_not_stale():
    """Every exemption names a parameter that is still required-and-defaultless."""
    defaults = _config_defaults()
    required = _schema_required()
    for name, reason in PER_RUN.items():
        assert reason.strip(), f"{name} has an empty reason"
        assert name in required, (
            f"PER_RUN names {name}, which the schema no longer requires"
        )
        assert name in defaults and defaults[name] is None, (
            f"PER_RUN names {name}, which nextflow.config no longer declares null -- "
            "it has a default now, so the exemption is dead"
        )


def test_the_template_tells_the_reader_how_to_use_it():
    """The template is only useful if it says what to copy it to and how to pass it."""
    text = TEMPLATE.read_text()
    assert "cp conf/site.config.template site.config" in text, (
        "the template does not show the copy command the docs quote verbatim"
    )
    assert "-c site.config" in text, (
        "the template does not show the `-c site.config` invocation it exists for"
    )
