"""Groovy/Nextflow source masking, shared by every guard that scans `.nf`/`.groovy` text.

`code_view(path)` returns the file with comment and string CONTENT blanked (offsets
preserved), so a naive `re.search(r"params\\.\\w+", ...)` -- or any other regex scan --
cannot mistake prose in a comment, or a filename substring inside a comment, for real
code. It masks `'...'`, `"..."`, `'''...'''`, triple-double-quoted, slashy `/.../` and
dollar-slashy `$/.../$` string forms, honouring backslash escapes everywhere except
`$/.../$`, where Groovy makes a backslash a literal character. GString interpolation is
not masked: `${...}` and `$ident` stay verbatim, so their braces still balance and a
read inside an interpolation is still counted.

Two safety nets catch a mis-tokenization: `_assert_balanced` and the
raw-newline-in-string check. Import `code_view` from here rather than writing a second
lexer.
"""

from __future__ import annotations

from pathlib import Path

_SLASHY_PRECEDERS = set("([{,;=~!&|?:+-*%<>^")

# Groovy allows a slashy string wherever an expression may START, and after a
# keyword the preceding character is alphanumeric (`return /re/`, `case /re/:`),
# so the classification looks at the preceding token, not just the character.
_SLASHY_KEYWORDS = frozenset(
    {
        "return",
        "case",
        "in",
        "when",
        "else",
        "assert",
        "new",
        "instanceof",
        "if",
        "while",
        "do",
        "switch",
        "throw",
        "yield",
        "and",
        "or",
        "not",
    }
)


def code_view(path: Path) -> str:
    """The file with comment and string CONTENT blanked, offsets preserved.

    Blanking in place (spaces, newlines kept) means every position found in the
    result still maps onto the real file, so failure messages quote real line
    numbers. GString interpolation -- `${...}` and `$ident.path` -- is kept
    verbatim: its braces must stay balanced with the code around them, and an
    identifier read inside an interpolation is a genuine read.
    """
    src = path.read_text()
    out = list(src)
    n = len(src)
    i = 0
    state = None  # None -> code; else {"q": terminator, "interp": bool}
    interp: list = []  # [[outer string state, brace depth], ...]

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    def prev_significant(pos: int):
        """(previous non-space char, the identifier ending at it or None)."""
        k = pos - 1
        while k >= 0 and out[k] in " \t\r\n":
            k -= 1
        if k < 0:
            return None, None
        ch = out[k]
        if not (ch.isalnum() or ch == "_"):
            return ch, None
        j = k
        while j >= 0 and (out[j].isalnum() or out[j] == "_"):
            j -= 1
        return ch, "".join(out[j + 1 : k + 1])

    while i < n:
        if state is None:
            c = src[i]
            if interp:
                # inside `${ ... }`: track depth so its closing brace is found
                if c == "{":
                    interp[-1][1] += 1
                    i += 1
                    continue
                if c == "}":
                    if interp[-1][1] == 0:
                        state = interp.pop()[0]
                        i += 1
                        continue
                    interp[-1][1] -= 1
                    i += 1
                    continue
            if src.startswith("//", i):
                j = src.find("\n", i)
                blank(i, n if j == -1 else j)
                i = n if j == -1 else j
                continue
            if src.startswith("/*", i):
                j = src.find("*/", i + 2)
                assert j != -1, f"{path}: unterminated block comment"
                blank(i, j + 2)
                i = j + 2
                continue
            if src.startswith("$/", i):
                # Dollar-slashy `$/ ... /$`: its body must not be scanned as code.
                state = {"q": "/$", "interp": True, "line": _line_of(src, i)}
                i += 2
                continue
            if src.startswith("'''", i) or src.startswith('"""', i):
                state = {
                    "q": src[i : i + 3],
                    "interp": c == '"',
                    "line": _line_of(src, i),
                }
                i += 3
                continue
            if c in "'\"":
                state = {"q": c, "interp": c == '"', "line": _line_of(src, i)}
                i += 1
                continue
            if c == "/":
                # Groovy's one genuine ambiguity: slashy string vs division.
                # Settled by the preceding significant token (punctuation or a
                # keyword that opens an expression). A heuristic: the balance and
                # raw-newline checks catch the cases it gets wrong.
                prev, word = prev_significant(i)
                if (
                    prev is None
                    or prev in _SLASHY_PRECEDERS
                    or word in _SLASHY_KEYWORDS
                ):
                    state = {"q": "/", "interp": True, "line": _line_of(src, i)}
                    i += 1
                    continue
            i += 1
            continue

        # --- inside a string literal ---
        q = state["q"]
        c = src[i]
        if c == "\\" and q != "/$":
            # `\` escapes in every Groovy string form EXCEPT `$/ ... /$`, where
            # it is a literal character and the escapes are `$$` and `$/`.
            blank(i, i + 2)
            i += 2
            continue
        if c == "\n" and len(q) == 1 and q != "/":
            # Groovy forbids a raw newline inside a '' or "" string, so the
            # opening quote was not a real string opener (usually a slashy
            # string misread as division). `_assert_balanced` cannot see this
            # case, so raise here.
            raise AssertionError(
                f"{path}:{state['line']}: a {q}-quoted string opened here is "
                f"still open at the end of line {_line_of(src, i)}, which Groovy "
                "does not allow. The "
                "scanner has mis-tokenized something before it (most likely a "
                "slashy string read as division), so this file's code view "
                "cannot be trusted. Refusing to audit rather than risk blanking "
                "a real read."
            )
        if src.startswith(q, i):
            blank(i, i + len(q))
            state = None
            i += len(q)
            continue
        if state["interp"] and src.startswith("${", i):
            interp.append([state, 0])  # `${` kept verbatim: braces must balance
            state = None
            i += 2
            continue
        if (
            state["interp"]
            and c == "$"
            and i + 1 < n
            and (src[i + 1].isalpha() or src[i + 1] == "_")
        ):
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] in "_."):
                j += 1
            i = j  # `$ident.path` kept verbatim: it is a real read
            continue
        blank(i, i + 1)
        i += 1

    assert state is None and not interp, (
        f"{path}: unterminated string literal -- refusing to audit this file "
        "rather than scan a truncated view of it."
    )
    code = "".join(out)
    _assert_balanced(path, code)
    return code


def _assert_balanced(path: Path, code: str) -> None:
    """The masked file's delimiters must balance, or the mask is wrong.

    If any string form was mis-tokenized, the delimiters it wrongly hid or
    exposed will almost certainly unbalance here, so the guard raises instead of
    auditing a truncated closure body.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list = []
    for i, c in enumerate(code):
        if c in "([{":
            stack.append((c, i))
        elif c in ")]}":
            assert stack and stack[-1][0] == pairs[c], (
                f"{path}:{_line_of(code, i)}: delimiters do not balance after "
                "comment/string masking, so the scan cannot be trusted. Refusing "
                "to audit rather than risk a truncated closure body."
            )
            stack.pop()
    assert not stack, (
        f"{path}:{_line_of(code, stack[0][1])}: unclosed {stack[0][0]!r} after "
        "comment/string masking, so the scan cannot be trusted."
    )


def line_of(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


# Alias for the leading-underscore spelling some importers use.
_line_of = line_of
