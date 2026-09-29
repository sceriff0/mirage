#!/usr/bin/env python3
"""A read-only copy of the pipeline AT A PINNED COMMIT, for arms that must run old code.

WHY. One benchmark compares methods that no longer coexist in one tree: STARE v1 (tier x
TRE-gate arms, legacy/robust SOLVE) was replaced by DRAPE on this branch -- its solvers,
`reg_tiled_gate_tre` and `reg_tiled_solver` are deleted, and both register as
`registration_method=tiled`. Rather than resurrect the old backend beside the new one, an
arm-plan row may carry `code_ref=<commit>`: run_arms.sh launches that row from a snapshot of
the repository at that commit, so it runs byte-for-byte the code that produced the results
already on disk under the same arm name -- which is what lets ARMS_RESUME=1 keep them.

WHY `git archive` AND NOT `git worktree add`. A snapshot is never edited and never
committed from; `git archive` writes plain files (modes kept, so bin/ scripts stay
executable), leaves no worktree bookkeeping in the source repository, and cannot be
switched to another branch by anyone -- the failure that mixed two branches' code in one
results root on 2026-09-29 (two heads, one checkout).

Layout: <root>/<full sha>/, with `.complete` written LAST, so an interrupted extraction is
never mistaken for a usable snapshot (it is removed and redone).

Usage::

    python3 -m benchmarks.code_snapshot --repo <checkout> --root <results>/.code <ref>
    # prints the snapshot directory
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ".complete"


class SnapshotError(RuntimeError):
    """The ref cannot be resolved or extracted; the message says what to do."""


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        raise SnapshotError(r.stderr.strip() or f"git {' '.join(args)} failed")
    return r.stdout.strip()


def resolve(repo: Path, ref: str) -> str:
    """`ref` -> full commit sha. Fetches from origin once if the commit is not local:
    a cluster checkout that never had the benchmarking branch checked out may lack it."""
    try:
        return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    except SnapshotError:
        pass
    try:
        _git(repo, "fetch", "--quiet", "origin")
    except SnapshotError as exc:
        raise SnapshotError(
            f"commit {ref!r} is not in {repo} and `git fetch origin` failed: {exc}"
        ) from exc
    try:
        return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    except SnapshotError as exc:
        raise SnapshotError(
            f"commit {ref!r} is not in {repo}, even after `git fetch origin`. "
            "Push the branch that holds it, or fix code_ref in arms.yaml."
        ) from exc


def materialise(repo: Path, ref: str, root: Path) -> Path:
    """The snapshot directory for `ref` under `root`, extracting it if needed. Idempotent."""
    # A plan names snapshots by full sha; a finished one needs no git at all (no fetch
    # from every launch, and it survives a checkout that has since lost the commit).
    done = root / ref / COMPLETE
    if len(ref) == 40 and done.is_file() and done.read_text().strip() == ref:
        return root / ref
    sha = resolve(repo, ref)
    dest = root / sha
    if (dest / COMPLETE).is_file():
        return dest
    if dest.exists():
        shutil.rmtree(dest)  # an interrupted extraction; never trust a partial tree
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".extract-") as tmp:
        tar = Path(tmp) / "src.tar"
        with open(tar, "wb") as fh:
            r = subprocess.run(
                ["git", "-C", str(repo), "archive", "--format=tar", sha],
                stdout=fh,
                stderr=subprocess.PIPE,
                check=False,
            )
        if r.returncode != 0:
            raise SnapshotError(
                f"git archive {sha} failed: {r.stderr.decode().strip()}"
            )
        stage = Path(tmp) / "tree"
        with tarfile.open(tar) as tf:
            try:
                tf.extractall(stage, filter="tar")
            except TypeError:  # Python < 3.10.12 has no extraction filters
                tf.extractall(stage)  # our own git archive: no foreign members
        (stage / COMPLETE).write_text(sha + "\n")
        stage.rename(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref", help="commit, tag or branch to snapshot")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--root", type=Path, required=True, help="e.g. <results>/.code")
    a = ap.parse_args(argv)
    try:
        print(materialise(a.repo, a.ref, a.root))
    except SnapshotError as exc:
        print(f"code_snapshot: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
