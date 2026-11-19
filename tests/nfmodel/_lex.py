"""A string-aware character walk over Nextflow/Groovy source.

tests/test_nfmodel.py covers the constructs a naive regex gets wrong, such as
a `stageAs: 'ref/*'` glob read as an open block comment, or a `{` inside a
comment mis-balancing a brace count.
"""

import re

__all__ = [
    "block_extent",
    "skip_non_code",
    "strip_comments",
    "strip_comments_and_strings",
]


def skip_non_code(text: str, i: int) -> int:
    """If `text[i:]` opens a `//` line comment, a `/* */` block comment, or a
    quoted string literal (single, double, or Groovy triple-quoted), return
    the index just past it. Otherwise return `i` unchanged.

    Shared by the block-extent walk and the comment/string stripper below, so
    a `{` or `}` -- or the word `publishDir` -- appearing inside a comment or
    a GString's `${...}` interpolation is treated as non-code exactly once,
    the same way, by both.
    """
    n = len(text)
    if text[i : i + 2] == "//":
        j = text.find("\n", i)
        return j if j != -1 else n
    if text[i : i + 2] == "/*":
        j = text.find("*/", i + 2)
        return j + 2 if j != -1 else n
    if text[i] in ("'", '"'):
        quote = text[i]
        if text[i : i + 3] == quote * 3:
            j = text.find(quote * 3, i + 3)
            return j + 3 if j != -1 else n
        j = i + 1
        while j < n and text[j] != quote:
            j += 2 if text[j] == "\\" else 1
        return min(j + 1, n)
    return i


def block_extent(text: str, start: int) -> int:
    """Given `start` just past a block's opening `{`, return the index of the
    matching closing `}` -- walking depth-first while skipping over comments
    and string literals, so a stray `{`/`}` inside either cannot mis-balance
    the walk.

    A naive brace count treats a `{` inside a comment as real nesting, runs
    past the block's own closing brace and borrows the next block's publishDir.
    """
    n = len(text)
    depth, i = 1, start
    while i < n and depth:
        skip_to = skip_non_code(text, i)
        if skip_to != i:
            i = skip_to
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i


def _strip(text: str, *, blank_strings: bool) -> str:
    """The one walk both stripping views share, via `skip_non_code`.

    The views differ only in what they do with a string literal's span: view A
    (`strip_comments_and_strings`) blanks it like a comment; view B
    (`strip_comments`) keeps it verbatim.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        skip_to = skip_non_code(text, i)
        if skip_to == i:
            out.append(text[i])
            i += 1
            continue
        span = text[i:skip_to]
        is_string = text[i] in ("'", '"')
        if is_string and not blank_strings:
            out.append(span)
        else:
            out.append(re.sub(r"[^\n]", " ", span))
        i = skip_to
    return "".join(out)


def strip_comments_and_strings(text: str) -> str:
    """Blank `//`/`/* */` comments and quoted-string contents to spaces
    (preserving newlines), so a `re.M`-anchored search over the result cannot
    match text that only appears in a comment or inside a string literal.

    So a block that only mentions `publishDir` in a comment does not satisfy
    `"publishDir" in block`.
    """
    return _strip(text, blank_strings=True)


def strip_comments(text: str) -> str:
    """Blank `//`/`/* */` comments only, keeping every string literal's
    contents -- quotes included -- verbatim.

    For guards that read text inside a string literal (e.g. `collectFile(name:
    'collated_versions.yml', sort: true)`), which `strip_comments_and_strings`
    would blank. Still string-aware, so a `stageAs: 'ref/*'` glob's `/*` is not
    read as an opening block comment.
    """
    return _strip(text, blank_strings=False)
