"""The freeze of the method code: per method, the fingerprint ``reuse_key`` carries as
``code`` and the files it covers (paths relative to the repository root).

Regenerate with ``python -m benchmarks.tests.regbench.make_method_code_freeze >
benchmarks/tests/regbench/method_code_freeze.json``, in a commit of its own that says why:
every cluster result of a method whose fingerprint moved runs again. A method added later
is appended (its own task); an existing entry is not rewritten to make a test pass."""

import json
import sys

from benchmarks.regbench import METHODS, methods


def build():
    out = {}
    for name in METHODS:
        files = sorted(
            str(p.relative_to(methods.REPO)) for p in methods.code_files(name)
        )
        out[name] = {"fingerprint": methods.code_fingerprint(name), "files": files}
    return out


if __name__ == "__main__":
    json.dump(build(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
