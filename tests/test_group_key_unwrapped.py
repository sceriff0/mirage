"""A `groupKey()` must never escape the `groupTuple()` it was created for.

`groupKey(id, size)` is a streaming size hint, not an identifier, but `groupTuple()`
emits the `GroupKey` object as element 0. Its `equals()` is asymmetric while its
hash matches the plain id::

    groupKey('P001', 2).equals('P001')  ==  true
    'P001'.equals(groupKey('P001', 2))  ==  false

so a `combine(by:)`/`join(by:)` that meets a GroupKey on one side and a String on
the other matches or silently drops depending on arrival order.

THE RULE
--------
A `groupTuple()` fed by a `groupKey()` must hand its result to a `.map {}` as the
very next operator, and that closure must either

  1. discard the key -- a leading-underscore parameter that is never read
     (`subworkflows/local/adapters/tiled_adapter.nf`), or
  2. unwrap every read -- `<param>.toString()`, `<param>[0].toString()`,
     `<param> as String` or `String.valueOf(<param>)`
     (`subworkflows/local/registration.nf`, `subworkflows/local/quantify_markers.nf`).

HOW IT IS CHECKED
-----------------
The operator chain is walked structurally from the closure holding `groupKey()`,
so the audited `.map {}` is the group's real consumer, never a nearby one.
`_code_view()` masks comments and string contents (all Groovy quote forms, slashy
and dollar-slashy) but keeps GString interpolation, whose reads still count.
Two nets catch a mis-mask: `_assert_balanced` raises on any bracket-count change,
and a raw newline inside a `'`/`"` string raises.

Known limits:
  1. Slashy-string detection is a heuristic (punctuation or `_SLASHY_KEYWORDS`
     before the `/`); an unlisted context is read as division.
  2. When a net fires, the guard refuses to audit and names the delimiter it
     choked on, which may be far from the leak.
  3. `$$` and `$/` escapes in dollar-slashy strings are not modelled.
  4. A grouped result assigned to a variable and used later is not traced; it
     fails the chain rule instead (loud, possibly a false positive).

The check is static because the failure is an arrival-order race that a
behavioural test passes by luck.
"""

from __future__ import annotations

import re
from pathlib import Path

from _code_view import code_view as _code_view
from _code_view import line_of as _line_of

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("subworkflows", "workflows", "modules", "tests")

_IDENT = r"[A-Za-z_]\w*"
_OPEN = "([{"
_CLOSE = ")]}"


def _match_delim(src: str, i: int) -> int:
    """Index of the delimiter closing the `(`/`[`/`{` at ``i``.

    Type-aware: a mismatched pair raises rather than silently pairing a brace
    with a bracket. Callers only ever pass positions taken from `_code_view()`,
    whose balance assertion has already run over the whole file.
    """
    assert src[i] in _OPEN, f"expected an opening delimiter at {i}, got {src[i]!r}"
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list = []
    for j in range(i, len(src)):
        if src[j] in _OPEN:
            stack.append(src[j])
        elif src[j] in _CLOSE:
            assert stack and stack[-1] == pairs[src[j]], "mismatched delimiters"
            stack.pop()
            if not stack:
                return j
    raise AssertionError("unbalanced delimiters")


def _skip_ws(src: str, i: int) -> int:
    while i < len(src) and src[i] in " \t\r\n":
        i += 1
    return i


def _enclosing_open_brace(src: str, pos: int) -> int | None:
    """Index of the innermost `{` that is still open at ``pos``."""
    depth = 0
    for i in range(pos - 1, -1, -1):
        c = src[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                return i
            depth -= 1
    return None


def _chain_ops(src: str, i: int) -> list[dict]:
    """Read the `.name(...)` / `.name {...}` operators that follow position ``i``.

    This is what makes the check structural: a Groovy statement never begins with
    `.`, so everything reachable this way is genuinely the same channel chain,
    and the chain stops exactly where the channel stops being piped onward.
    """
    ops: list[dict] = []
    while True:
        j = _skip_ws(src, i)
        if j >= len(src) or src[j] != ".":
            return ops
        m = re.match(_IDENT, src[j + 1 :])
        if not m:
            return ops
        name = m.group(0)
        k = _skip_ws(src, j + 1 + len(name))
        op = {"name": name, "pos": j, "kind": None, "open": None, "close": None}
        if k < len(src) and src[k] in "({":
            close = _match_delim(src, k)
            op["kind"] = src[k]
            op["open"] = k
            op["close"] = close
            i = close + 1
        else:
            i = j + 1 + len(name)
        ops.append(op)


def _closure_params(src: str, open_brace: int, close_brace: int):
    """(parameter names, body) of a `{ a, b -> ... }` closure, or (None, body).

    The parameter list is only recognised when the text before the first `->`
    holds nothing but identifiers and commas -- otherwise that arrow belongs to
    a nested closure and this one has no declared parameters.
    """
    inner = src[open_brace + 1 : close_brace]
    arrow = inner.find("->")
    if arrow == -1:
        return None, inner
    head = inner[:arrow]
    if not re.fullmatch(r"[\sA-Za-z_,\w]*", head) or any(c in head for c in "(){}[];="):
        return None, inner
    params = [p.strip() for p in head.split(",") if p.strip()]
    return (params or None), inner[arrow + 2 :]


def _group_key_sites() -> list[dict]:
    """Every real `groupKey(...)`, resolved to the chain it belongs to.

    Each record carries the `groupKey()` line, the operators that follow its
    closure, and -- when the chain is well formed -- the consuming `.map {}`'s
    parameters and body. `problem` is set when the chain itself is wrong, in
    which case there is nothing left to audit inside a closure.
    """
    sites: list[dict] = []
    for d in SCAN_DIRS:
        for path in sorted((REPO_ROOT / d).rglob("*.nf")):
            src = _code_view(path)
            for m in re.finditer(r"\bgroupKey\s*\(", src):
                rel = path.relative_to(REPO_ROOT)
                site = {
                    "path": rel,
                    "line": _line_of(src, m.start()),
                    "params": None,
                    "body": None,
                    "map_line": None,
                    "problem": None,
                }
                sites.append(site)
                where = f"{rel}:{site['line']}"

                brace = _enclosing_open_brace(src, m.start())
                if brace is None or _match_delim(src, brace) < m.start():
                    site["problem"] = (
                        f"{where}: this groupKey() is not inside a closure, so the "
                        "chain it belongs to cannot be resolved. Build the key in a "
                        ".map {} that feeds the groupTuple()."
                    )
                    continue

                ops = _chain_ops(src, _match_delim(src, brace) + 1)
                names = [o["name"] for o in ops]
                if "groupTuple" not in names:
                    site["problem"] = (
                        f"{where}: this groupKey() does not feed a groupTuple() in its "
                        f"own chain (operators after it: {names or 'none'}). A groupKey "
                        "outside a groupTuple() is a GroupKey wrapper with no purpose."
                    )
                    continue

                gt = names.index("groupTuple")
                if gt == len(ops) - 1:
                    site["problem"] = (
                        f"{where}: the groupTuple() at "
                        f"{rel}:{_line_of(src, ops[gt]['pos'])} ends its chain, so the "
                        "nextflow.extension.GroupKey wrapper leaves the grouping as the "
                        "group key and reaches whatever consumes this channel -- a "
                        "process input, an emit, another operator. That is the bug this "
                        "guard exists for. Consume it with a .map {} that unwraps or "
                        "discards the key."
                    )
                    continue

                nxt = ops[gt + 1]
                if nxt["name"] != "map" or nxt["kind"] != "{":
                    site["problem"] = (
                        f"{where}: the groupTuple() at "
                        f"{rel}:{_line_of(src, ops[gt]['pos'])} is followed by "
                        f".{nxt['name']}(...) at {rel}:{_line_of(src, nxt['pos'])}, not "
                        "by a .map {} -- so nothing unwraps the GroupKey before it "
                        "travels on. Put the unwrapping .map {} directly after the "
                        "groupTuple()."
                    )
                    continue

                params, body = _closure_params(src, nxt["open"], nxt["close"])
                site["map_line"] = _line_of(src, nxt["pos"])
                site["body"] = body
                if not params:
                    site["problem"] = (
                        f"{where}: the .map {{}} consuming this groupTuple() at "
                        f"{rel}:{site['map_line']} declares no parameters, so the group "
                        "key is reached as `it[0]` and cannot be audited. Name the "
                        "parameters."
                    )
                    continue
                site["params"] = params
    return sites


# Every accepted way of turning the GroupKey back into a plain String. A GString
# (`"${k}"`) is not here: GString.equals(String) is false too, so
# interpolation swaps one asymmetric key for another.
# `.toString()`, `?.toString()` (safe navigation) and `k .toString()` (spaced).
_CALL_TOSTRING = r"\s*\??\s*\.\s*toString\(\)"
_UNWRAP_TAIL = re.compile(
    _CALL_TOSTRING  # k.toString()
    + r"|\s*\[\s*0\s*\]"
    + _CALL_TOSTRING  # k[0].toString()  (single-parameter closure)
    + r"|\s+as\s+String\b"  # k as String
)


def _leaking_reads(key: str, body: str) -> list[re.Match]:
    reads = []
    for mo in re.finditer(rf"\b{re.escape(key)}\b", body):
        tail = body[mo.end() :]
        if _UNWRAP_TAIL.match(tail):
            continue
        if body[: mo.start()].endswith("String.valueOf("):
            continue
        # A map-literal key (`[patient_id: pid]`) is a label, not a read, but a
        # ternary's then-branch (`cond ? group_key : x`) has the same tail and is
        # a leak. A map key follows `[`, `,`, `(` or the body start; a then-branch
        # follows `?`.
        head = body[: mo.start()].rstrip()
        if re.match(r"\s*:(?!:)", tail) and (head == "" or head[-1] in "[,("):
            continue
        if mo.start() and body[mo.start() - 1] == ".":
            continue  # a property of something else that shares the name
        reads.append(mo)
    return reads


def test_leaking_reads_tells_a_map_key_from_a_ternary():
    """The map-literal skip must not swallow a ternary's then-branch.

    `[k: v]` is a label and must be skipped -- `quantify_markers.nf`'s
    `patient_id: pid` entry depends on it, and deleting the skip fails there.
    `cond ? k : v` wears the same `name` + `:` tail and is a real leak. Both
    directions are pinned so neither can regress into the other.
    """
    # map-literal keys and named arguments: skipped
    assert _leaking_reads("k", "[k: 1]") == []
    assert _leaking_reads("k", "def m = [\n    id: x,\n    k: y,\n]") == []
    assert _leaking_reads("k", "foo(k: 1)") == []
    assert _leaking_reads("k", "k: 1") == []
    # ternary then-branch: a genuine leak, on one line and split over three
    assert len(_leaking_reads("k", "def id = n > 1 ? k : other")) == 1
    assert len(_leaking_reads("k", "def id = n > 1\n    ? k\n    : other")) == 1


def test_group_key_sites_are_found():
    """The scan must actually see the groupKey call sites it claims to guard.

    A guard whose glob matches nothing passes silently forever; this pins the
    inventory so deleting or moving one is a deliberate act. Adding a genuinely
    new groupKey() site means updating this list AND satisfying the escape check
    below -- extending the list alone does not make a leak pass.
    """
    sites = _group_key_sites()
    files = sorted({str(s["path"]) for s in sites})
    assert files == [
        "subworkflows/local/adapters/tiled_adapter.nf",
        "subworkflows/local/quantify_markers.nf",
        "subworkflows/local/registration.nf",
    ], f"groupKey() call sites moved; scan found {files}"
    assert len(sites) == 4, f"expected 4 groupKey() sites, found {len(sites)}"


def test_group_key_never_escapes_its_group_tuple():
    """Each grouping closure discards the key, or reads it only unwrapped."""
    problems = []
    for site in _group_key_sites():
        if site["problem"]:
            problems.append(site["problem"])
            continue
        key = site["params"][0]
        where = f"{site['path']}:{site['line']}"
        reads = _leaking_reads(key, site["body"])
        if key.startswith("_"):
            if reads:
                problems.append(
                    f"{where}: the groupTuple() key is named `{key}` at "
                    f"{site['path']}:{site['map_line']}, i.e. declared as discarded, but "
                    f"is read {len(reads)} time(s) in that closure. Rename it and unwrap "
                    "with .toString(), or stop reading it."
                )
        elif reads:
            problems.append(
                f"{where}: the groupTuple() key `{key}` at "
                f"{site['path']}:{site['map_line']} is read {len(reads)} time(s) WITHOUT "
                ".toString(), so the nextflow.extension.GroupKey wrapper escapes the "
                "grouping. GroupKey.equals(String) is true but String.equals(GroupKey) "
                "is false while their hashCodes match, so a downstream "
                "combine(by:)/join(by:) against a plain-String key matches in one "
                f"direction only and silently drops tuples depending on arrival order. "
                f"Unwrap it: `def id = {key}.toString()`."
            )
    assert not problems, "GroupKey escapes its grouping:\n  " + "\n  ".join(problems)
