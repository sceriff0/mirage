"""Static guard: DEEPCELL_ACCESS_TOKEN's VALUE must never be interpolated into
Nextflow config text.

A GString such as
``"--env DEEPCELL_ACCESS_TOKEN=${System.getenv('DEEPCELL_ACCESS_TOKEN')}"`` in
`conf/modules.config`'s SEGMENT `containerOptions` would bake the secret's value
into the rendered `.command.run`, which is group-readable on a shared HPC
filesystem and survives `-resume`.

The token is forwarded by name only: Docker via `-e DEEPCELL_ACCESS_TOKEN`,
Singularity via `singularity.envWhitelist = 'DEEPCELL_ACCESS_TOKEN'`
(`nextflow.config`, `singularity` profile).
`tests/modules/segment_deepcell_token.nf.test` is the execution-level check;
this one is static so it runs in the plain `python-tests` job. It enforces
SECURITY.md's "Secrets & Site Configuration" policy for this token.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES_CONFIG = ROOT / "conf" / "modules.config"
NEXTFLOW_CONFIG = ROOT / "nextflow.config"

TOKEN = "DEEPCELL_ACCESS_TOKEN"

# The vulnerable pattern: the value-interpolating GString literal,
# `"--env DEEPCELL_ACCESS_TOKEN=${System.getenv('DEEPCELL_ACCESS_TOKEN')}"`.
# Matches on either quote style Groovy allows around the env var name.
VALUE_INTERPOLATION_RE = re.compile(
    re.escape(TOKEN) + r"=\$\{",
)
GETENV_INSIDE_GSTRING_RE = re.compile(
    r"\$\{[^}]*System\.getenv\(\s*['\"]" + re.escape(TOKEN) + r"['\"]\s*\)[^}]*\}"
)

SINGULARITY_PROFILE_RE = re.compile(r"\bsingularity\s*\{")
ENV_WHITELIST_RE = re.compile(
    r"singularity\.envWhitelist\s*=\s*['\"]" + re.escape(TOKEN) + r"['\"]"
)


def _extract_singularity_profile_block(config_text: str) -> str:
    """Return the raw text inside the `singularity { ... }` profile block."""
    m = SINGULARITY_PROFILE_RE.search(config_text)
    assert m, "Could not locate a `singularity { ... }` block in nextflow.config"

    brace_start = config_text.index("{", m.start())
    depth = 0
    for i in range(brace_start, len(config_text)):
        ch = config_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return config_text[brace_start + 1 : i]

    raise AssertionError("Unclosed `singularity { ... }` block in nextflow.config")


def test_modules_config_never_interpolates_the_token_value():
    """`conf/modules.config` must never bake DEEPCELL_ACCESS_TOKEN's value in.

    Two independent checks against the same regression, so a partial fix (or
    a differently-shaped reintroduction) is still caught:

    1. The literal vulnerable shape: `DEEPCELL_ACCESS_TOKEN=${` immediately
       after the flag (`--env DEEPCELL_ACCESS_TOKEN=${...}`).
    2. More generally: `System.getenv('DEEPCELL_ACCESS_TOKEN')` must never
       appear *inside* a `${...}` GString interpolation anywhere in the
       file -- that is what turns the getenv() call's return value into
       literal text written to `.command.run`. Using it outside a GString
       (e.g. as a Groovy boolean/truthiness guard, `... && System.getenv(...)`)
       is fine.
    """
    text = MODULES_CONFIG.read_text()

    assert not VALUE_INTERPOLATION_RE.search(text), (
        f"conf/modules.config contains the vulnerable value-interpolation "
        f"pattern '{TOKEN}=${{' -- the secret's value would be written "
        f"literally into .command.run. Forward by name only (Docker: "
        f"'-e {TOKEN}'; Singularity: singularity.envWhitelist)."
    )
    assert not GETENV_INSIDE_GSTRING_RE.search(text), (
        f"conf/modules.config interpolates System.getenv('{TOKEN}') inside a "
        f'GString ("...${{...}}...") -- this bakes the secret\'s value into '
        f"the rendered config/command text. Use the getenv() call only as a "
        f"boolean guard (e.g. '&& System.getenv(...)'), never inside ${{}}."
    )


def test_nextflow_config_singularity_profile_declares_env_whitelist():
    """`nextflow.config`'s `singularity` profile must forward the token by name.

    `singularity.envWhitelist = 'DEEPCELL_ACCESS_TOKEN'` is what lets
    conf/modules.config's containerOptions contribute nothing for the token
    on the Singularity branch -- Nextflow itself forwards it by name,
    equivalent to Docker's `-e VAR`, reading the value from the environment
    at run time rather than baking it into any generated file.
    """
    text = NEXTFLOW_CONFIG.read_text()
    singularity_block = _extract_singularity_profile_block(text)

    assert ENV_WHITELIST_RE.search(singularity_block), (
        f"nextflow.config's `singularity` profile no longer declares "
        f"singularity.envWhitelist = '{TOKEN}'. Without it, "
        f"conf/modules.config's SEGMENT containerOptions has no way to "
        f"forward the token to the Singularity container by name, and "
        f"CellSAM auto-download would silently break (or someone would be "
        f"tempted to reintroduce value interpolation to fix it)."
    )
