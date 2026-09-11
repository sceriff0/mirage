"""Query interface over the repo's Nextflow source.

Guards assert against this model, never against raw text. `processes()` and
`with_name_blocks()` both drive their pattern matches off the
comment/string-stripped text produced by `strip_comments_and_strings` --
never off the raw source -- so a comment that merely *mentions* a process
name or a `withName: 'X' {` selector cannot be counted as the real thing.
`script:`/`stub:` bodies are the one deliberate exception: they are sliced
from the RAW text, because the reason to read them is usually to inspect the
rendered command, and the stripped view has blanked every string in it.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ._lex import block_extent, strip_comments, strip_comments_and_strings

REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_DIRS = ("modules", "subworkflows", "workflows")
_PROCESS_RE = re.compile(r"^\s*process\s+([A-Z][A-Z0-9_]*)\s*\{", re.M)
_WITHNAME_RE = re.compile(r"withName:\s*'([^']+)'\s*\{")
_PARAM_RE = re.compile(r"\bparams\.([A-Za-z_]\w*)")
_WORKFLOW_RE = re.compile(r"^\s*workflow\s+([A-Z][A-Z0-9_]*)\s*\{", re.M)
# `include { NAME as ALIAS }` -- the alias is the name the pipeline (and
# `withName:`) actually sees at the call site.
_ALIAS_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+as\s+([A-Z][A-Z0-9_]*)")


@dataclass(frozen=True)
class Process:
    name: str
    path: Path
    body: str
    raw_body: str
    script_body: str
    stub_body: str
    inputs: str
    outputs: str


@dataclass(frozen=True)
class WithNameBlock:
    selector: str
    names: List[str]
    body: str
    raw_body: str
    start_line: int


def nf_files(dirs: Sequence[str] = _DEFAULT_DIRS, root: Path = REPO_ROOT) -> List[Path]:
    """Every .nf file under `dirs`, recursively (`rglob`) and sorted for a
    deterministic order. A non-recursive `glob('*.nf')` misses e.g. a nested
    `modules/local/sub/*.nf` -- this is the enumerator that used to be wrong."""
    out: List[Path] = []
    for d in dirs:
        out.extend(sorted((root / d).rglob("*.nf")))
    return out


def nf_test_files(root: Path = REPO_ROOT) -> List[Path]:
    """Every nf-test file under `tests/`, recursively and sorted for a
    deterministic order. A guard scanning `tests/**/*.nf.test` on its own
    (a raw `Path.glob`) is itself an undiscoverable private parse of Nextflow
    source by `test_nfmodel.py`'s own `_discover_nf_source_readers` rule --
    this is the one place that enumeration happens."""
    return sorted((root / "tests").rglob("*.nf.test"))


def _section(body: str, name: str) -> str:
    """Text of a `name:` section, up to the next top-level section keyword
    (`input`/`output`/`when`/`script`/`shell`/`exec`/`stub`), or to the end
    of `body` if `name` is the last section present."""
    keys = ("input", "output", "when", "script", "shell", "exec", "stub")
    m = re.search(rf"^\s*{name}\s*:", body, re.M)
    if not m:
        return ""
    rest = body[m.end() :]
    nxt = [
        n.start() for k in keys for n in [re.search(rf"^\s*{k}\s*:", rest, re.M)] if n
    ]
    return rest[: min(nxt)] if nxt else rest


@lru_cache(maxsize=None)
def processes(root: Path = REPO_ROOT) -> Dict[str, Process]:
    """Every Nextflow process under `modules/`, keyed by process name.

    Matches are found on the comment/string-stripped text so a comment that
    only mentions `process FOO {` cannot be counted. `strip_comments_and_strings`
    preserves length and line count exactly, so the match offsets found on
    the stripped text are valid offsets into the raw text too -- that is what
    lets `raw_body` come back with its strings and comments intact even
    though the match that located it could not have been fooled by one.
    """
    out: Dict[str, Process] = {}
    for f in nf_files(("modules",), root):
        raw = f.read_text()
        clean = strip_comments_and_strings(raw)
        for m in _PROCESS_RE.finditer(clean):
            start = m.end()
            end = block_extent(clean, start)
            raw_body, body = raw[start:end], clean[start:end]
            out[m.group(1)] = Process(
                name=m.group(1),
                path=f,
                raw_body=raw_body,
                body=body,
                # script:/stub: bodies come from the RAW text -- the point of
                # reading them is usually to inspect the rendered command, and
                # the stripped view has blanked every string in it.
                script_body=_section(raw_body, "script"),
                stub_body=_section(raw_body, "stub"),
                inputs=_section(body, "input"),
                outputs=_section(body, "output"),
            )
    return out


@lru_cache(maxsize=None)
def with_name_blocks(root: Path = REPO_ROOT) -> List[WithNameBlock]:
    """Every `withName: '...' {` block in `conf/modules.config`.

    The selector regex needs its literal quotes, but `strip_comments_and_strings`
    blanks a string literal's delimiting quotes along with its contents (not
    just comments) -- so matching it against the stripped text directly finds
    nothing at all. Instead, match against the RAW text (which has real
    quotes to match), then keep only matches whose start position is
    unchanged between raw and stripped text: a real selector's `withName:`
    text survives stripping untouched, while one that only appears inside a
    `//` or `/* */` comment -- e.g. "the `withName: 'SEGMENT'` ext.args
    closure below" -- is blanked to spaces there and gets filtered out.
    """
    path = root / "conf" / "modules.config"
    raw = path.read_text()
    clean = strip_comments_and_strings(raw)
    out: List[WithNameBlock] = []
    for m in _WITHNAME_RE.finditer(raw):
        if clean[m.start()] != raw[m.start()]:
            continue
        start = m.end()
        end = block_extent(clean, start)
        out.append(
            WithNameBlock(
                selector=m.group(1),
                names=m.group(1).split("|"),
                raw_body=raw[start:end],
                body=clean[start:end],
                start_line=raw.count("\n", 0, m.start()) + 1,
            )
        )
    return out


@lru_cache(maxsize=None)
def workflows(root: Path = REPO_ROOT) -> Dict[str, Path]:
    """Every named `workflow NAME {` under `workflows/` and `subworkflows/`.

    A subworkflow is a first-class pipeline name -- SEG_QC, VALIS_ADAPTER,
    REGISTER_PATIENT -- and is indistinguishable from a process to anything
    reading prose or a diagram. A guard that resolves UPPER_SNAKE_CASE names
    against `processes()` alone therefore reports every subworkflow as
    nonexistent. Matched on the comment/string-stripped view, so a subworkflow
    named only in a comment does not count as declared.

    The anonymous entry `workflow {` in `main.nf` has no name and is not here.
    """
    out: Dict[str, Path] = {}
    for f in nf_files(("workflows", "subworkflows"), root):
        for m in _WORKFLOW_RE.finditer(strip_comments_and_strings(f.read_text())):
            out[m.group(1)] = f
    return out


@lru_cache(maxsize=None)
def include_aliases(root: Path = REPO_ROOT) -> Dict[str, str]:
    """`alias -> original` for every `include { X as Y }` in the pipeline.

    An aliased include creates a name that exists nowhere as a `process` or
    `workflow` declaration -- SEG_QC_SEGMENT is `SEGMENT` imported under
    another name -- yet it is the name that appears in the trace, in
    `withName:` selectors and in every diagram. Resolving names without it
    reports a real, running task as nonexistent.
    """
    out: Dict[str, str] = {}
    for f in nf_files(("modules", "workflows", "subworkflows"), root) + [
        root / "main.nf"
    ]:
        clean = strip_comments_and_strings(f.read_text())
        for m in re.finditer(r"include\s*\{([^}]*)\}", clean):
            for a in _ALIAS_RE.finditer(m.group(1)):
                out[a.group(2)] = a.group(1)
    return out


def script_bodies(root: Path = REPO_ROOT) -> Dict[str, str]:
    """process name -> its script: body (comments/strings intact, see
    `processes()`)."""
    return {n: p.script_body for n, p in processes(root).items()}


def param_refs(text: str) -> set:
    """`params.<name>` reads found in `text`. A match immediately followed by
    `(` is a `Map` method call (e.g. `params.subMap(...)`), not a parameter
    read, and is excluded. Callers that want comments excluded too must strip
    them first -- this function does not know about comments."""
    found = set()
    for m in _PARAM_RE.finditer(text):
        if text[m.end() :].lstrip().startswith("("):
            continue
        found.add(m.group(1))
    return found


@dataclass(frozen=True)
class NfTestCase:
    path: Path
    name: str
    tags: frozenset
    stub_option: bool
    process: Optional[str]
    reads_command_sh: bool


# nf-test accepts both quote styles everywhere, so every one of these matches
# `['"]` and backreferences the opening quote. `-stub-run` is Nextflow's
# documented primary spelling and `-stub` its alias: a case written the primary
# way used to model as `stub_option=False`, i.e. as a RENDERED case, which is
# how a stub-only case could satisfy the rendered-coverage rule in
# test_process_nf_test_coverage.py while rendering nothing.
_NF_TEST_CASE_RE = re.compile(r"""\btest\s*\(\s*(?P<q>["'])(?P<name>.*?)(?P=q)\s*\)\s*\{""")
_NF_TEST_TAG_RE = re.compile(r"""^\s*tag\s+(?P<q>["'])(?P<val>[^"']+)(?P=q)""", re.M)
_NF_TEST_PROCESS_RE = re.compile(
    r"""^\s*process\s+(?P<q>["'])(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P=q)""", re.M
)
_NF_TEST_STUB_OPTION_RE = re.compile(
    r"""^\s*options\s+(?P<q>["'])[^"']*(?<![\w-])-stub(?:-run)?(?![\w-])[^"']*(?P=q)""", re.M
)


def _tag_values(text: str) -> set:
    return {m.group("val") for m in _NF_TEST_TAG_RE.finditer(text)}


@lru_cache(maxsize=None)
def nf_test_cases(root: Path = REPO_ROOT) -> List[NfTestCase]:
    """Every `test("...") { ... }` case in every nf-test file under `tests/`.

    Parsed on the comment-stripped text, so a `tag "stub"` inside a comment is
    not a tag and a `test("x")` inside a comment is not a case. Strings are
    kept, because every name, tag and option IS a string. A file-preamble tag
    (before the first case) applies to every case in the file; a case's own
    tags are unioned in. `process` is the file's `process "NAME"` directive,
    None for workflow/pipeline files. `reads_command_sh` says whether the case
    body mentions `.command.sh` at all -- the difference between a case that
    asserts the RENDERED command and one (e.g. a pure `process.failed` case)
    that merely happens to run without `-stub`.
    """
    out: List[NfTestCase] = []
    for f in nf_test_files(root):
        clean = strip_comments(f.read_text())
        pm = _NF_TEST_PROCESS_RE.search(clean)
        process = pm.group("name") if pm else None
        first = _NF_TEST_CASE_RE.search(clean)
        preamble = clean[: first.start()] if first else clean
        file_tags = _tag_values(preamble)
        for cm in _NF_TEST_CASE_RE.finditer(clean):
            start = cm.end()
            body = clean[start : block_extent(clean, start)]
            out.append(
                NfTestCase(
                    path=f,
                    name=cm.group("name"),
                    tags=frozenset(file_tags | _tag_values(body)),
                    stub_option=bool(_NF_TEST_STUB_OPTION_RE.search(body)),
                    process=process,
                    reads_command_sh=".command.sh" in body,
                )
            )
    return out
