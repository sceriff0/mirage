#!/usr/bin/env python3
"""Guard: CI cache keys must be keyed on MANIFESTS, and each cached path must have
an owner in the file that caches it.

A cache defect is invisible in a green run: a cache only changes how long the answer
takes. The rules:

  1. **No source-tree hash in a cache key.** ``~/.nextflow`` is the launcher and
     framework jars for one Nextflow version; hashing ``**/*.nf`` into its key misses on
     every pipeline commit and fills the 10 GB repository LRU with never-reused entries.
  2. **A cached path has an owner in the caching file, and its key names what
     determines its bytes.** ``nf-core/setup-nf-test`` caches ``~/.nf-test`` itself;
     caching it again elsewhere under a different key makes its install fail with "dest
     already exists" on an nf-test version bump.
  3. **One key, one cache step.** A cache entry is immutable (first writer wins), so two
     steps sharing a key race.

Scope is ``tests/ci_actions.scanned_files()``: every workflow and every local composite
action, where the repository's ``actions/cache`` steps live. The rules are written out in
``.github/actions/setup-nextflow/action.yml``. Both tables below are closed: a new cached
path or hashed glob fails until someone records what it is for.

Plain pytest, paths derived from ``__file__``, per this directory's other guards.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
ci_actions = importlib.import_module("ci_actions")


# ---------------------------------------------------------------------------
# The two tables. Both CLOSED: anything absent from them fails.
# ---------------------------------------------------------------------------

# Every cached path declares TWO things, and both are checked:
#
#   installer        -- a substring that must appear in the SAME file. A path may
#                       be cached only where something puts the bytes there.
#   key_must_contain -- a substring that must appear in that cache step's `key:`.
#                       Without the Nextflow version in the distribution key, the
#                       matrix legs would share one immutable entry.
CACHED_PATH_OWNERS = {
    "~/.nextflow": (
        "nf-core/setup-nextflow",
        "inputs.nextflow-version",
        "the Nextflow launcher + framework jars, written by the install step; the "
        "VERSION is the whole of what determines them, so the key must name it",
    ),
    "${{ github.workspace }}/.nextflow-plugins": (
        "nextflow plugin install",
        "hashFiles('nextflow.config')",
        "NXF_PLUGINS_DIR, filled by the plugin provisioning step; nextflow.config "
        "declares the pin, so its hash is what determines the contents",
    ),
}

# Every glob whose content is hashed into a cache key. Rule 1: manifests only.
HASHED_GLOBS = {
    "nextflow.config": (
        "the file that DECLARES the nf-schema plugin version -- a manifest. It "
        "rotates the plugin cache exactly when the pin changes."
    ),
    "requirements/**": (
        "the Python pin files, read by CI and by every container image alike -- "
        "a manifest tree. `cache: 'pip'` would otherwise default to "
        "`**/requirements.txt`, which matches only the root shim and "
        "docs/requirements.txt and never rotates when a pin changes."
    ),
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
_HASHFILES_RE = re.compile(r"hashFiles\(\s*'([^']*)'\s*\)")


def _steps_of_file(path: Path):
    """Every step in a file, workflow or composite action, as (where, step)."""
    data = yaml.safe_load(path.read_text()) or {}
    rel = path.relative_to(ROOT).as_posix()
    out = []
    for job_id, job in (data.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                out.append((f"{rel}::{job_id}", step))
    for step in (data.get("runs") or {}).get("steps") or []:
        if isinstance(step, dict):
            out.append((rel, step))
    return out


def cache_steps():
    """Every `actions/cache` step across workflows and composite actions."""
    found = []
    for path in ci_actions.scanned_files():
        for where, step in _steps_of_file(path):
            if "actions/cache" in str(step.get("uses", "")):
                found.append((path, where, step))
    return found


def _uncommented(text: str) -> str:
    """`text` with its `#` comments removed -- whole-line AND trailing.

    The owner check is a substring search, so a comment naming the owner (whole-line
    or trailing) must not satisfy it. Uses the quote-aware
    `ci_actions.strip_line_comment`, which does not truncate `VERSION="${VERSION#v}"`.
    """
    return "\n".join(
        stripped
        for stripped in (
            ci_actions.strip_line_comment(line) for line in text.splitlines()
        )
        if stripped.strip()
    )


def _lines(value) -> list[str]:
    """A `path:`/`restore-keys:` value, single-line or block scalar, as a list."""
    return [
        line.strip()
        for line in str(value or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def hashed_globs():
    """Every glob whose content ends up in a cache key, as (where, field, glob).

    Two forms, and this repo writes both:

      * `hashFiles('<glob>')` inside an `actions/cache` `key:`/`restore-keys:`;
      * `cache-dependency-path:` (actions/setup-python) or
        `cache-dependency-glob:` (astral-sh/setup-uv), which the action hashes
        into a key it builds itself.
    """
    out = []
    for path, where, step in cache_steps():
        with_block = step.get("with") or {}
        for field in ("key", "restore-keys"):
            for glob in _HASHFILES_RE.findall(str(with_block.get(field, ""))):
                out.append((where, field, glob))
    for path in ci_actions.scanned_files():
        for where, step in _steps_of_file(path):
            with_block = step.get("with") or {}
            for field in ("cache-dependency-path", "cache-dependency-glob"):
                for glob in _lines(with_block.get(field)):
                    out.append((where, field, glob))
    return out


# ---------------------------------------------------------------------------
# Non-vacuity: every check below is a "must not find" scan.
# ---------------------------------------------------------------------------
def test_the_scan_found_the_caches_this_repository_has():
    """A scan that reads zero cache steps would pass every rule below having checked
    nothing."""
    steps = cache_steps()
    assert len(steps) >= 2, (
        f"found {len(steps)} `actions/cache` step(s) across "
        f"{[p.relative_to(ROOT).as_posix() for p in ci_actions.scanned_files()]}. "
        "This repository caches the Nextflow distribution and the Nextflow plugins "
        "directory; if that is no longer true, delete this file with them, and if it "
        "is, the `uses: actions/cache` match or the file scope broke."
    )
    globs = hashed_globs()
    assert len(globs) >= 2, (
        f"found {len(globs)} hashed glob(s) ({globs}). The plugin cache hashes "
        "nextflow.config and the pip cache hashes requirements/**; a smaller number "
        "means this scan is not reading the keys it thinks it is."
    )


# ---------------------------------------------------------------------------
# Rule 1: hashFiles() targets manifests, never a source tree.
# ---------------------------------------------------------------------------
def test_no_cache_key_hashes_anything_but_a_declared_manifest():
    """A source-tree hash guarantees a MISS on every commit that matters.

    The entry is still saved, so every such run adds a never-reused entry to the
    10 GB repository-wide LRU and evicts caches that do get hits. The table is
    closed: a new glob fails here until someone records why it is a manifest.
    """
    offenders = []
    for where, field, glob in hashed_globs():
        if glob not in HASHED_GLOBS:
            offenders.append(f"{where}: `{field}` hashes {glob!r}")
    assert not offenders, (
        "a cache key hashes a glob that is not a declared manifest:\n  "
        + "\n  ".join(offenders)
        + "\n\nhashFiles() targets MANIFESTS (requirements/*, nextflow.config, a "
        "lockfile), never a source tree: a source hash misses on every commit that "
        "changes the thing being built and writes a fresh, never-reused entry each "
        "time. If this really is a manifest, add it to HASHED_GLOBS with the reason.\n"
        "Declared manifests: " + ", ".join(sorted(HASHED_GLOBS))
    )


# ---------------------------------------------------------------------------
# Rule 2, first half: a cached path must have an installer in the same file.
# ---------------------------------------------------------------------------
def test_every_cached_path_has_a_declared_installer_in_the_same_file():
    """A cached path must be created by something in the same file (rule 2).

    ``nf-core/setup-nf-test`` does
    install ``~/.nf-test``, but it lives in a different composite action, so a
    cache of that path in ``setup-nextflow`` is unowned there and fails here even
    if someone adds a CACHED_PATH_OWNERS entry for it.
    """
    offenders = []
    checked = 0
    for path, where, step in cache_steps():
        text = _uncommented(path.read_text())
        for cached in _lines((step.get("with") or {}).get("path")):
            checked += 1
            if cached not in CACHED_PATH_OWNERS:
                offenders.append(
                    f"{where}: caches {cached!r}, which is not in CACHED_PATH_OWNERS"
                )
                continue
            owner, _key_component, reason = CACHED_PATH_OWNERS[cached]
            if owner not in text:
                offenders.append(
                    f"{where}: caches {cached!r} but {owner!r} ({reason}) does not "
                    f"appear in {path.relative_to(ROOT).as_posix()}, so nothing in "
                    "that file puts the bytes there"
                )
    assert checked >= 2, (
        f"only {checked} cached path(s) examined; the scan is not reading them"
    )
    assert not offenders, (
        "cached path(s) with no owner in the file that caches them:\n  "
        + "\n  ".join(offenders)
        + "\n\nCache the directory where it is INSTALLED, or not at all. A second "
        "entry for bytes another action already caches is keyed on the wrong thing "
        "and eventually restores a file that action then refuses to overwrite."
    )


# ---------------------------------------------------------------------------
# Rule 2, second half: one key, one cache.
# ---------------------------------------------------------------------------
def test_no_two_cache_steps_share_a_key():
    """A cache entry is immutable: first writer wins, later saves silently no-op.

    Two steps caching different paths under one key template is therefore a race
    whose outcome is whichever job finished first -- and it is silent, because
    both steps report a cache hit forever after.
    """
    by_key: dict[str, list[str]] = {}
    for _path, where, step in cache_steps():
        key = " ".join(str((step.get("with") or {}).get("key", "")).split())
        assert key, f"{where}: `actions/cache` step with no `key:` at all"
        by_key.setdefault(key, []).append(where)
    clashes = {k: v for k, v in by_key.items() if len(v) > 1}
    assert not clashes, (
        "cache step(s) share a key template:\n  "
        + "\n  ".join(f"{k!r}: {v}" for k, v in sorted(clashes.items()))
        + "\n\nThe first save under a key wins and every later one silently no-ops, "
        "so the two caches are not sharing -- one of them is being discarded."
    )


# ---------------------------------------------------------------------------
# restore-keys says whether the key can miss.
# ---------------------------------------------------------------------------
def test_restore_keys_exist_exactly_where_the_key_can_miss():
    """A prefix fallback is meaningful for one kind of key and pointless for the
    other, and this repository has one of each.

    * A key carrying a CONTENT HASH (the plugins cache, hashing nextflow.config)
      misses whenever the manifest changes, which is precisely when a warm start
      is still mostly valid -- so it needs a prefix to fall back to.
    * A key fully determined by a VERSION STRING (the Nextflow distribution)
      either hits or the version was bumped, and a bumped version wants a fresh
      download, not the previous version's jars. A prefix there restores the very
      entry the key already names, or something deliberately superseded.

    """
    offenders = []
    for _path, where, step in cache_steps():
        with_block = step.get("with") or {}
        key = str(with_block.get("key", ""))
        has_hash = bool(_HASHFILES_RE.search(key))
        has_fallback = bool(_lines(with_block.get("restore-keys")))
        if has_hash and not has_fallback:
            offenders.append(
                f"{where}: key hashes a manifest but declares no `restore-keys:`, so "
                "every pin change is a fully cold start"
            )
        if not has_hash and has_fallback:
            offenders.append(
                f"{where}: key has no content hash, so it hits or the version moved -- "
                "a `restore-keys:` prefix can only restore the entry the key already "
                "names, or a superseded one"
            )
    assert not offenders, "\n".join(offenders)


def test_every_cached_path_is_keyed_on_what_determines_its_contents():
    """The other half of rule 2: a key names everything its bytes depend on.

    `test_no_two_cache_steps_share_a_key` catches two separate steps colliding;
    this catches one step whose key lost a component and collides with itself
    across matrix legs.
    """
    offenders = []
    checked = 0
    for _path, where, step in cache_steps():
        with_block = step.get("with") or {}
        key = str(with_block.get("key", ""))
        for cached in _lines(with_block.get("path")):
            if cached not in CACHED_PATH_OWNERS:
                continue  # test_every_cached_path_has_a_declared_installer... owns this
            _owner, component, reason = CACHED_PATH_OWNERS[cached]
            checked += 1
            if component not in key:
                offenders.append(
                    f"{where}: caches {cached!r} under key {key!r}, which does not "
                    f"contain {component!r} -- {reason}"
                )
    assert checked >= 2, (
        f"only {checked} cached path(s) had a declared key component; the scan is "
        "not reading them"
    )
    assert not offenders, (
        "cache key(s) do not name what determines the bytes they cover:\n  "
        + "\n  ".join(offenders)
        + "\n\nA cache entry is IMMUTABLE. A key missing a component merges two "
        "different caches into one entry and the first writer wins -- silently, "
        "and forever, because both steps report a hit afterwards."
    )


def test_every_workflow_that_installs_nextflow_pins_nxf_plugins_dir():
    """`NXF_PLUGINS_DIR` keeps the distribution cache a pure function of the version.

    The distribution key is os/arch/version, which is only correct while the plugins
    live elsewhere. Without the `env:`, nf-schema would land in `~/.nextflow/plugins`
    and be frozen into a version-only entry that never rotates when `nextflow.config`
    changes the pin.

    Scope is every JOB that reaches `.github/actions/setup-nextflow`, resolved
    through `tests/ci_actions.py`, so a job that picks it up via `setup-nf-test`
    counts too. `env:` may be declared at workflow or job level; both work.
    """
    var = "NXF_PLUGINS_DIR"
    offenders = []
    checked = 0
    for path in ci_actions.workflow_files():
        data = yaml.safe_load(path.read_text()) or {}
        workflow_env = data.get("env") or {}
        for job_id, job in (data.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            actions = {p.parent.name for p in ci_actions.job_local_actions(job)}
            if "setup-nextflow" not in actions:
                continue
            checked += 1
            job_env = job.get("env") or {}
            if var not in workflow_env and var not in job_env:
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}::{job_id} installs Nextflow "
                    f"but neither it nor its workflow sets {var}"
                )
    assert checked >= 3, (
        f"only {checked} job(s) were found to install Nextflow. ci.yml, release.yml "
        "and nightly.yml all do, several times over; a number this small means the "
        "resolver stopped reaching .github/actions/setup-nextflow."
    )
    assert not offenders, (
        f"{var} must be set wherever Nextflow is installed -- without it the plugins "
        "land inside ~/.nextflow and get frozen into a cache key that only names the "
        "Nextflow version:\n  " + "\n  ".join(offenders)
    )
