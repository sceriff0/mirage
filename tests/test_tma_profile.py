"""The `tma` profile: a data-shape profile that composes with a site profile.

Tissue-microarray cores are small (a 1 mm core is ~2000-3000 px on the long side at
typical scan resolutions), and VALIS 1.0.0-1.2.0 dies on any slide whose full
resolution is no larger than its non-rigid registration size -- the `high` tier's
4096 px (bin/utils/valis_preflight.py has the chain; REGISTER now refuses such input
at start). The remedy is `memory_mode = 'custom'` with a small
`reg_valis_max_non_rigid_dim`, and a profile is the right vehicle: it says what the
DATA is, the way `ieo` says where the run happens, and the two compose
(`-profile slurm,ieo,tma`). A profile pin reaches these two parameters because both
are read late -- by ParamUtils.validateRegPresets in workflow scope and by REGISTER's
script block at task time -- never frozen at config-parse time the way the publishDir
gates were (tests/test_cleanup_publish_gates.py).

1024 px is the `low` tier's non-rigid size and is safe for any core of 2048 px or
more on its long side, with the margin VALIS's tissue-mask term needs. A CLI
`--reg_valis_max_non_rigid_dim` still outranks the profile for larger cores.
"""

from __future__ import annotations

import re

from tests.nfmodel import REPO_ROOT, strip_comments

CONFIG = strip_comments((REPO_ROOT / "nextflow.config").read_text())
NON_RIGID_PX = 1024


def _profile_body(name: str) -> str:
    m = re.search(rf"^\s{{4}}{name}\s*\{{(.*?)^\s{{4}}\}}", CONFIG, re.S | re.M)
    assert m, f"no `{name}` profile in nextflow.config"
    return m.group(1)


def test_the_tma_profile_pins_exactly_the_custom_pair():
    body = _profile_body("tma")
    assigned = dict(re.findall(r"params\.([a-z_]+)\s*=\s*([^\n]+)", body))
    assert assigned == {
        "memory_mode": "'custom'",
        "reg_valis_max_non_rigid_dim": str(NON_RIGID_PX),
    }, assigned


def test_the_tma_size_is_the_low_tiers_non_rigid_size():
    """Not an arbitrary number: the `low` row's non-rigid size, the smallest value the
    tier table already vouches for."""
    presets = (REPO_ROOT / "bin" / "utils" / "valis_config.py").read_text()
    low = re.search(r'"low":\s*\{(.*?)\}', presets, re.S).group(1)
    low_nr = int(
        re.search(r'"max_non_rigid_registration_dim_px":\s*(\d+)', low).group(1)
    )
    assert NON_RIGID_PX == low_nr


def test_the_profile_is_in_the_parse_sweep():
    """tests/check_profiles_parse.sh parses every shipped profile; a profile it does
    not know is a profile nothing checks."""
    sweep = (REPO_ROOT / "tests" / "check_profiles_parse.sh").read_text()
    m = re.search(r'PROFILES="\$\{1:-([^}]+)\}"', sweep)
    assert m and "tma" in m.group(1).split(","), m.group(0) if m else "no PROFILES line"


def test_the_profile_is_documented_where_operators_look():
    usage = (REPO_ROOT / "docs" / "usage.md").read_text()
    params = (REPO_ROOT / "docs" / "parameters.md").read_text()
    assert "-profile" in usage and "tma" in usage, (
        "docs/usage.md does not mention the tma profile"
    )
    assert "tma" in params and "slurm,ieo,tma" in params, (
        "docs/parameters.md's Tiers section does not show the profile composing with a site profile"
    )
