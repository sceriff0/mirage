"""The download scripts (``benchmarks/regbench/download/``) with a stand-in ``curl``,
``zip`` and ``unzip`` on PATH: nothing is downloaded.

Checked: every URL of the URL file is fetched to its name; a bad md5 exits non-zero and
leaves the ``.part`` and no final file; without the URL file the script refuses before
any ``curl``; ANHIR's split archive is rejoined and both archives unzipped where
``prepare anhir`` reads them; HyReCo fetches the core zip only unless
``HYRECO_ADDITIONAL=1``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DL = ROOT / "benchmarks" / "regbench" / "download"

CURL = """#!/usr/bin/env bash
# stand-in curl: writes "payload:<url>" to the -o file and logs the url
out=""; url=""
while [ $# -gt 0 ]; do
  case $1 in -o) out=$2; shift 2;; -C|--retry) shift 2;; -*) shift;; *) url=$1; shift;; esac
done
echo "$url" >> "$LOG_DIR/curl.log"
printf 'payload:%s' "$url" > "$out"
"""
ZIP = """#!/usr/bin/env bash
echo "zip $*" >> "$LOG_DIR/zip.log"
for a in "$@"; do last=$a; done
touch "$last"
"""
UNZIP = """#!/usr/bin/env bash
echo "unzip $*" >> "$LOG_DIR/unzip.log"
d=""; prev=""
for a in "$@"; do [ "$prev" = "-d" ] && d=$a; prev=$a; done
mkdir -p "$d/HyReCo/HE" "$d/HyReCo/PHH3"
"""


def _md5(text):
    return hashlib.md5(text.encode()).hexdigest()


@pytest.fixture
def env(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in (("curl", CURL), ("zip", ZIP), ("unzip", UNZIP)):
        p = bin_ / name
        p.write_text(body)
        p.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    e = {k: v for k, v in os.environ.items() if k not in ("ANHIR_URLS", "HYRECO_URLS")}
    e.update(PATH=f"{bin_}:{os.environ['PATH']}", LOG_DIR=str(logs))
    e.pop("HYRECO_ADDITIONAL", None)
    return e, logs, tmp_path


def _run(script, dest, env, **extra):
    return subprocess.run(
        ["bash", str(DL / script), str(dest)],
        env={**env, **extra},
        capture_output=True,
        text=True,
    )


def _log(logs, name):
    p = logs / f"{name}.log"
    return p.read_text().split("\n")[:-1] if p.exists() else []


ANHIR_NAMES = (
    "dataset_medium.csv",
    *(f"dataset_medium.z0{i}" for i in range(1, 6)),
    "dataset_medium.zip",
    "landmarks.zip",
)


def _url_file(path, names, md5=None):
    lines = ["# the links", ""]
    for n in names:
        url = f"https://example.invalid/{n}"
        lines.append("\t".join([url, n] + ([md5[n]] if md5 and n in md5 else [])))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_anhir_fetches_every_url_and_unpacks_where_prepare_reads(env):
    e, logs, tmp = env
    urls = _url_file(
        tmp / "anhir.tsv",
        ANHIR_NAMES,
        md5={"landmarks.zip": _md5("payload:https://example.invalid/landmarks.zip")},
    )
    r = _run("anhir.sh", tmp / "anhir", e, ANHIR_URLS=str(urls))
    assert r.returncode == 0, r.stderr
    assert _log(logs, "curl") == [f"https://example.invalid/{n}" for n in ANHIR_NAMES]
    for n in ANHIR_NAMES:
        assert (tmp / "anhir" / n).is_file() and not (
            tmp / "anhir" / f"{n}.part"
        ).exists()
    (zip_call,) = _log(logs, "zip")
    assert zip_call.startswith("zip -s 0 ") and zip_call.endswith(
        "dataset_medium_joined.zip"
    )
    unz = _log(logs, "unzip")
    assert any(
        "dataset_medium_joined.zip" in u and u.endswith("/anhir/images") for u in unz
    )
    assert any("landmarks.zip" in u and u.endswith("/anhir/landmarks") for u in unz)
    # run again: nothing is fetched twice
    r = _run("anhir.sh", tmp / "anhir", e, ANHIR_URLS=str(urls))
    assert r.returncode == 0 and len(_log(logs, "curl")) == len(ANHIR_NAMES)


def test_a_bad_md5_fails_and_leaves_the_part(env):
    e, logs, tmp = env
    urls = _url_file(
        tmp / "anhir.tsv", ANHIR_NAMES, md5={"dataset_medium.csv": "0" * 32}
    )
    r = _run("anhir.sh", tmp / "anhir", e, ANHIR_URLS=str(urls))
    assert r.returncode != 0
    assert "md5" in r.stderr
    assert (tmp / "anhir" / "dataset_medium.csv.part").is_file()
    assert not (tmp / "anhir" / "dataset_medium.csv").exists()
    assert _log(logs, "zip") == [] and _log(logs, "unzip") == []


@pytest.mark.parametrize(
    "script,var", [("anhir.sh", "ANHIR_URLS"), ("hyreco.sh", "HYRECO_URLS")]
)
def test_refuses_without_the_url_file_before_any_curl(env, script, var):
    e, logs, tmp = env
    r = _run(script, tmp / "d", e)
    assert r.returncode == 2 and var in r.stderr
    r = _run(script, tmp / "d", e, **{var: str(tmp / "missing.tsv")})
    assert r.returncode != 0
    assert _log(logs, "curl") == []


def test_anhir_refuses_a_url_file_without_the_cover_table(env):
    e, logs, tmp = env
    urls = _url_file(
        tmp / "anhir.tsv", [n for n in ANHIR_NAMES if n != "dataset_medium.csv"]
    )
    r = _run("anhir.sh", tmp / "anhir", e, ANHIR_URLS=str(urls))
    assert r.returncode == 2 and "dataset_medium.csv" in r.stderr
    assert _log(logs, "curl") == []


HYRECO_NAMES = ("HyReCo.zip", "HyReCo-Additional-HE.zip", "HyReCo-Additional-PHH3.zip")


def test_hyreco_fetches_the_core_zip_only_by_default(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES)
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls))
    assert r.returncode == 0, r.stderr
    assert _log(logs, "curl") == ["https://example.invalid/HyReCo.zip"]
    (unz,) = _log(logs, "unzip")
    assert "HyReCo/HE/*" in unz and "HyReCo/PHH3/*" in unz
    assert "free space of" in r.stdout and "466.7 GB" in r.stdout


def test_hyreco_additional_fetches_every_zip(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES)
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls), HYRECO_ADDITIONAL="1")
    assert r.returncode == 0, r.stderr
    assert _log(logs, "curl") == [f"https://example.invalid/{n}" for n in HYRECO_NAMES]
    assert len(_log(logs, "unzip")) == 3


def test_hyreco_additional_refuses_a_url_file_without_the_additional_zips(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES[:1])
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls), HYRECO_ADDITIONAL="1")
    assert r.returncode == 2 and "HyReCo-Additional-HE.zip" in r.stderr
    assert _log(logs, "curl") == []


def test_no_failure_is_swallowed():
    for p in DL.glob("*.sh"):
        assert "|| true" not in p.read_text(), p
