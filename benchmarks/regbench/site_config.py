"""Where the StarDist re-run of the bare nuclei gets its model and settings from.

The cohort's published outlines were made by the pipeline with the SITE's StarDist model
and whatever the site overlay sets (``conf/ieo.config`` on the cluster checkout; it is
gitignored and never part of the repository). A re-run that is to reproduce them must
use the same values, so they are resolved here, in this order, per value:

1. the environment: ``REGBENCH_STARDIST_MODEL``, ``REGBENCH_STARDIST_MODEL_DIR`` and
   ``REGBENCH_STARDIST_PROB_THRESH`` (explicit variables always win);
2. a site config: ``REGBENCH_SITE_CONFIG`` when set (``none`` reads no file), else
   ``<repo>/conf/ieo.config`` when it exists;
3. the repository default (``nextflow.config``), for the segmentation settings only. The
   model has no usable default: the name ``nextflow.config`` ships is not a StarDist
   built-in and needs a folder.

Only these keys are read from a site config, as plain ``key = literal`` lines:
``segmentation_model``, ``segmentation_model_dir`` and the settings ``bin/segment.py`` is
rendered with (``SETTING_KEYS``). The file is read comment-blind (``//`` and ``/* */``
outside quotes). A key assigned twice with different values (two profiles) is refused
rather than guessed. The file is read, never written, and no path from it is stored
anywhere but the records of the run that used it.

Standard library only.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SITE_CONFIG = REPO / "conf" / "ieo.config"
ENV_SITE_CONFIG = "REGBENCH_SITE_CONFIG"
ENV_MODEL = "REGBENCH_STARDIST_MODEL"
ENV_MODEL_DIR = "REGBENCH_STARDIST_MODEL_DIR"
ENV_PROB_THRESH = "REGBENCH_STARDIST_PROB_THRESH"
NO_FILE = ("none", "-")
MODEL_KEYS = ("segmentation_model", "segmentation_model_dir")
# the settings conf/modules.config renders into `segment.py` for StarDist
SETTING_KEYS = (
    "seg_n_tiles_y",
    "seg_n_tiles_x",
    "seg_expand_distance",
    "seg_pmin",
    "seg_pmax",
    "seg_prob_thresh",
)
KEYS = MODEL_KEYS + SETTING_KEYS
ENV_OF = {
    "segmentation_model": ENV_MODEL,
    "segmentation_model_dir": ENV_MODEL_DIR,
    "seg_prob_thresh": ENV_PROB_THRESH,
}
FROM_DEFAULT = "repository default (nextflow.config)"
NOT_SET = "not set"
_ASSIGN = re.compile(r"^\s*(?:params\.)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$")
_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def strip_comments(text):
    """``text`` without ``//`` and ``/* */`` comments; quoted strings are left alone."""
    out, i, n, quote = [], 0, len(text), ""
    while i < n:
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            out.append(" ")
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def literal(raw):
    """A config literal as a Python value: a quoted string, a number, ``null``,
    ``true`` / ``false``. Anything else (an expression, a closure) raises ValueError."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        if "${" in raw and raw[0] == '"':
            raise ValueError("an interpolated string")
        return raw[1:-1]
    if raw == "null":
        return None
    if raw in ("true", "false"):
        return raw == "true"
    if _NUMBER.fullmatch(raw):
        return float(raw) if re.search(r"[.eE]", raw) else int(raw)
    raise ValueError("not a plain literal")


def assignments(text, keys=KEYS):
    """``{key: [raw value, ...]}`` of the plain ``key = value`` lines of ``keys``."""
    found = {}
    for line in strip_comments(text).splitlines():
        m = _ASSIGN.match(line)
        if m and m.group(1) in keys:
            found.setdefault(m.group(1), []).append(m.group(2))
    return found


def read(path, keys=KEYS):
    """``{key: value}`` for the keys a config file sets. A key set more than once with
    different values, or to something that is not a literal, is refused."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise SystemExit(f"site config {path}: cannot be read ({exc})") from None
    out = {}
    for key, raws in assignments(text, keys).items():
        values = []
        for raw in raws:
            try:
                value = literal(raw)
            except ValueError as exc:
                raise SystemExit(
                    f"site config {path}: {key} = {raw} is {exc}; it cannot be read "
                    "here." + _way_out(key)
                ) from None
            if value not in values:
                values.append(value)
        if len(values) > 1:
            raise SystemExit(
                f"site config {path}: {key} is set {len(raws)} times, to "
                f"{', '.join(repr(v) for v in values)}; which one the arms ran with "
                "cannot be told from the file." + _way_out(key)
            )
        out[key] = values[0]
    return out


def _way_out(key):
    env = ENV_OF.get(key)
    if env:
        return f" Export {env} to say it."
    return (
        f" Point {ENV_SITE_CONFIG} at a config that sets it once (or at `none` for the "
        "repository default)."
    )


def repository_defaults(repo=None):
    """The settings' defaults, read from the repository's ``nextflow.config``."""
    return read(Path(repo or REPO) / "nextflow.config", SETTING_KEYS)


def site_config_path(env=None, default=None):
    """The site config to read, or None: ``REGBENCH_SITE_CONFIG`` (which must exist;
    ``none`` or ``-`` means no file), else ``conf/ieo.config`` of the checkout if it is
    there."""
    env = os.environ if env is None else env
    named = (env.get(ENV_SITE_CONFIG) or "").strip()
    if named:
        if named.lower() in NO_FILE:
            return None
        if not Path(named).is_file():
            raise SystemExit(f"{ENV_SITE_CONFIG}={named}: no such file")
        return Path(named)
    default = Path(DEFAULT_SITE_CONFIG if default is None else default)
    return default if default.is_file() else None


def resolve(env=None, repo=None, default=None):
    """Every value of the re-run and where it is from::

        {"site_config": "<path>" | "",
         "values": {key: {"value": v, "source": text, "default": d, "differs": bool}}}

    ``default`` / ``differs`` are there for the settings only (against
    ``nextflow.config``). A model that nothing names has value ``""`` and source
    ``not set``."""
    env = os.environ if env is None else env
    path = site_config_path(env, default)
    site = read(path) if path is not None else {}
    defaults = repository_defaults(repo)
    from_site = f"site config ({path})"
    values = {}
    for key in MODEL_KEYS:
        named = (env.get(ENV_OF[key]) or "").strip()
        if named:
            values[key] = {"value": named, "source": f"environment ({ENV_OF[key]})"}
        elif site.get(key) not in (None, ""):
            values[key] = {"value": str(site[key]), "source": from_site}
        else:
            values[key] = {"value": "", "source": NOT_SET}
    for key in SETTING_KEYS:
        base = defaults.get(key)
        value, source = base, FROM_DEFAULT
        if key in site:
            value, source = site[key], from_site
        named = (env.get(ENV_OF.get(key, "")) or "").strip() if key in ENV_OF else ""
        if named:  # kept as written: it is rendered into the command as it is
            value, source = named, f"environment ({ENV_OF[key]})"
        values[key] = {
            "value": value,
            "source": source,
            "default": base,
            "differs": value != base,
        }
    return {"site_config": str(path) if path is not None else "", "values": values}


def lines(resolved):
    """``name<TAB>value<TAB>source`` per value, for a log or a dry run."""
    out = []
    for key, v in resolved["values"].items():
        value = "null" if v["value"] is None else str(v["value"])
        note = ""
        if v.get("differs"):
            note = f" [differs from the repository default {v['default']}]"
        out.append(f"{key}\t{value}\t{v['source']}{note}")
    return out
