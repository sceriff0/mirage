import json
from pathlib import Path

from benchmarks.regbench import methods

FREEZE = Path(__file__).with_name("method_code_freeze.json")


def test_no_frozen_method_changed():
    frozen = json.loads(FREEZE.read_text())
    moved = {m: v for m, v in frozen.items() if methods.code_fingerprint(m) != v["fingerprint"]}
    assert not moved, (
        f"frozen method code changed: {sorted(moved)} (files: "
        f"{sorted({f for v in moved.values() for f in v['files']})}). A frozen file changes "
        "only in a commit of its own that regenerates method_code_freeze.json "
        "(python -m benchmarks.tests.regbench.make_method_code_freeze) and says why: every "
        "cluster result of that method will run again."
    )
