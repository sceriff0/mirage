"""The `tma` profile: a data-shape profile that composes with a site profile.

Tissue-microarray cores are small (a 1 mm core is ~2000-3000 px on the long side at
typical scan resolutions), and VALIS 1.0.0-1.2.0 dies on any slide whose full
resolution is no larger than its non-rigid registration size -- the `high` tier's
2048 px (bin/utils/valis_preflight.py has the chain; REGISTER now refuses such input
at start). The remedy is `memory_mode = 'custom'` with a small
`reg_valis_max_non_rigid_dim`, and a profile is the right vehicle: it says what the
DATA is, the way `ieo` says where the run happens, and the two compose
(`-profile slurm,ieo,tma`). A profile pin reaches these two parameters because both
are read late -- by ParamUtils.validateRegPresets in workflow scope and by REGISTER's
script block at task time -- never frozen at config-parse time the way the publishDir
gates were (tests/test_cleanup_publish_gates.py).

1024 px is the `medium` tier's size and is safe for any core of 2048 px or
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


REGISTER_GB = 64
# A flat heap, not register.nf's 32 + 16 x attempt ramp: Bio-Formats on five ~2800 px
# cores needs single-digit GiB, and the ramp's min(48, task.memory - 4) would hand Java
# most of a small request -- the Python side, where REGISTER's peak (SuperGlue matching)
# actually is, gets the remainder.
JVM_HEAP_GB = 16


def test_the_tma_profile_pins_exactly_the_custom_pair_and_the_jvm_heap():
    body = _profile_body("tma")
    assigned = dict(re.findall(r"params\.([a-z_]+)\s*=\s*([^\n]+)", body))
    assert assigned == {
        "memory_mode": "'custom'",
        "reg_valis_max_non_rigid_dim": str(NON_RIGID_PX),
        "reg_jvm_heap_gb": str(JVM_HEAP_GB),
    }, assigned


def test_the_jvm_pin_is_below_the_ramps_attempt_1_request():
    """The pin exists to hand memory BACK to Python under the profile's request. If it
    is not smaller than what the ramp would have derived on attempt 1
    (min(48, REGISTER_GB - 4)), it changes nothing."""
    assert JVM_HEAP_GB < min(48, REGISTER_GB - 4)


def test_the_tma_size_is_the_medium_tiers_non_rigid_size():
    """Not an arbitrary number: the `medium` row's size, a value the tier table already
    vouches for (the `tma` profile keeps `high`'s 2048 px feature matching and lowers
    only the non-rigid stage)."""
    presets = (REPO_ROOT / "bin" / "utils" / "valis_config.py").read_text()
    low = re.search(r'"medium":\s*\{(.*?)\}', presets, re.S).group(1)
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


def test_the_tma_profile_sizes_register_for_cores_not_slides():
    """conf/modules.config reserves a flat 300 GB x attempt for REGISTER, sized for
    whole slides. Five ~2800 px cores need a small fraction of that, and on SLURM a
    300 GB request is what the run waits in the queue for. The profile overrides the
    one process, keeps the retry ramp, and leaves cpus/time alone; the JVM heap is
    pinned flat by the profile (JVM_HEAP_GB above), not derived from task.memory."""
    body = _profile_body("tma")
    at = body.find("withName: 'REGISTER'")
    assert at != -1, "no withName: 'REGISTER' override in the tma profile"
    # Everything from the selector to the profile's end: the selector's own block
    # ends at the first `}` past the closure, so a non-greedy brace match would stop
    # inside the closure. Nothing else follows the override in this profile.
    override = body[at:]
    assert re.search(
        rf"memory\s*=\s*\{{\s*{REGISTER_GB}\.GB \* task\.attempt\s*\}}", override
    ), override
    assert "cpus" not in override and "time" not in override, (
        "the profile should override memory only; cpus and time stay with conf/modules.config"
    )


def test_the_register_override_is_documented_next_to_the_300_gb_row():
    resources = (REPO_ROOT / "docs" / "resources.md").read_text()
    assert "tma" in resources and f"{REGISTER_GB} GB" in resources, (
        "docs/resources.md's REGISTER row does not say the tma profile lowers it"
    )
