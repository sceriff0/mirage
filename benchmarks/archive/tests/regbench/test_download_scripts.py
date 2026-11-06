"""The download script (``benchmarks/regbench/download/hyreco.sh``) with a stand-in
``curl`` and ``unzip`` on PATH: nothing is downloaded.

Checked: every URL the script wants is fetched to its name; a bad md5 exits non-zero and
leaves the ``.part`` and no final file; without the URL file the script refuses before
any ``curl``; HyReCo fetches the core zip AND both Additional zips by default (user ruling
2026-10-19: "I need full set and additional"), the core zip alone with
``HYRECO_ADDITIONAL=0``; the free-space hint is the sum over what it fetches.
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
# stand-in unzip: each zip unpacks to its own layout (the core zip to HyReCo/{HE,PHH3},
# each Additional zip to HyReCo-Additional/<stain>/ with one file in it); UNZIP_TOP_HE /
# UNZIP_TOP_PHH3 name another top-level folder for that Additional zip (a zip whose
# layout is not the one prepare reads). `-Z1` lists the zip's entries and logs nothing
UNZIP = """#!/usr/bin/env bash
d=""; prev=""; zip=""; list=0
for a in "$@"; do
  [ "$prev" = "-d" ] && d=$a
  case $a in -Z1) list=1;; *.zip) [ -n "$zip" ] || zip=$a;; esac
  prev=$a
done
case $(basename "$zip") in
  HyReCo-Additional-HE.zip) top=${UNZIP_TOP_HE:-HyReCo-Additional}; stains=HE; file=1;;
  HyReCo-Additional-PHH3.zip) top=${UNZIP_TOP_PHH3:-HyReCo-Additional}; stains=PHH3; file=1;;
  *) top=HyReCo; stains="HE PHH3"; file=0;;
esac
if [ "$list" = 1 ]; then
  for s in $stains; do echo "$top/$s/"; echo "$top/$s/1.tif"; done
  exit 0
fi
echo "unzip $*" >> "$LOG_DIR/unzip.log"
for s in $stains; do
  mkdir -p "$d/$top/$s"
  [ "$file" = 0 ] || : > "$d/$top/$s/1.tif"
done
"""


def _md5(text):
    return hashlib.md5(text.encode()).hexdigest()


@pytest.fixture
def env(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in (("curl", CURL), ("unzip", UNZIP)):
        p = bin_ / name
        p.write_text(body)
        p.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    e = {k: v for k, v in os.environ.items() if k != "HYRECO_URLS"}
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


def _url_file(path, names, md5=None):
    lines = ["# the links", ""]
    for n in names:
        url = f"https://example.invalid/{n}"
        lines.append("\t".join([url, n] + ([md5[n]] if md5 and n in md5 else [])))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("script,var", [("hyreco.sh", "HYRECO_URLS")])
def test_refuses_without_the_url_file_before_any_curl(env, script, var):
    e, logs, tmp = env
    r = _run(script, tmp / "d", e)
    assert r.returncode == 2 and var in r.stderr
    r = _run(script, tmp / "d", e, **{var: str(tmp / "missing.tsv")})
    assert r.returncode != 0
    assert _log(logs, "curl") == []


HYRECO_NAMES = ("HyReCo.zip", "HyReCo-Additional-HE.zip", "HyReCo-Additional-PHH3.zip")


def test_hyreco_fetches_every_zip_by_default(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES)
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls))
    assert r.returncode == 0, r.stderr
    assert _log(logs, "curl") == [f"https://example.invalid/{n}" for n in HYRECO_NAMES]
    unz = _log(logs, "unzip")
    assert len(unz) == 3
    assert "HyReCo/HE/*" in unz[0] and "HyReCo/PHH3/*" in unz[0]
    # the hint: twice each zip, summed over the three (233.36 + 273.73 + 232.58 GB)
    assert "free space of" in r.stdout and "1479.3 GB" in r.stdout
    for n in HYRECO_NAMES:
        assert (tmp / "h" / n).is_file() and not (tmp / "h" / f"{n}.part").exists()
    # run again: nothing is fetched twice
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls))
    assert r.returncode == 0 and len(_log(logs, "curl")) == len(HYRECO_NAMES)


def test_hyreco_additional_0_fetches_the_core_zip_only(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES[:1])
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls), HYRECO_ADDITIONAL="0")
    assert r.returncode == 0, r.stderr
    assert _log(logs, "curl") == ["https://example.invalid/HyReCo.zip"]
    (unz,) = _log(logs, "unzip")
    assert "HyReCo/HE/*" in unz and "HyReCo/PHH3/*" in unz
    assert "free space of" in r.stdout and "466.7 GB" in r.stdout
    assert "1479.3 GB" not in r.stdout


def test_hyreco_fails_when_subset_b_is_not_where_prepare_reads_it(env):
    """Final review I2: after the unzip, HyReCo-Additional/HE and /PHH3 must be there and
    hold something; else the script fails naming the top-level folders the two
    Additional zips really hold."""
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES)
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls), UNZIP_TOP_HE="Additional_HE")
    assert r.returncode == 1, r.stdout
    assert "HyReCo-Additional/HE" in r.stderr and "missing or empty" in r.stderr
    assert "HyReCo-Additional-HE.zip: Additional_HE" in r.stderr
    assert "HyReCo-Additional-PHH3.zip: HyReCo-Additional" in r.stderr
    assert "done; next" not in r.stdout
    # an empty folder is not there either
    (tmp / "h2" / "HyReCo-Additional" / "PHH3").mkdir(parents=True)
    r = _run("hyreco.sh", tmp / "h2", e, HYRECO_URLS=str(urls), UNZIP_TOP_PHH3="elsewhere")
    assert r.returncode == 1 and "HyReCo-Additional/PHH3" in r.stderr
    assert "HyReCo-Additional-PHH3.zip: elsewhere" in r.stderr


def test_hyreco_additional_0_never_checks_subset_b(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES[:1])
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls), HYRECO_ADDITIONAL="0")
    assert r.returncode == 0, r.stderr
    assert not (tmp / "h" / "HyReCo-Additional").exists()


def test_hyreco_refuses_a_url_file_without_the_additional_zips_by_default(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES[:1])
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls))
    assert r.returncode == 2 and "HyReCo-Additional-HE.zip" in r.stderr
    assert "HYRECO_ADDITIONAL=0" in r.stderr
    assert _log(logs, "curl") == []


def test_a_bad_md5_fails_and_leaves_the_part(env):
    e, logs, tmp = env
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES, md5={"HyReCo.zip": "0" * 32})
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls))
    assert r.returncode != 0
    assert "md5" in r.stderr
    assert (tmp / "h" / "HyReCo.zip.part").is_file()
    assert not (tmp / "h" / "HyReCo.zip").exists()
    assert _log(logs, "unzip") == []


def test_a_good_md5_is_accepted(env):
    e, logs, tmp = env
    good = _md5("payload:https://example.invalid/HyReCo.zip")
    urls = _url_file(tmp / "hyreco.tsv", HYRECO_NAMES, md5={"HyReCo.zip": good})
    r = _run("hyreco.sh", tmp / "h", e, HYRECO_URLS=str(urls))
    assert r.returncode == 0, r.stderr
    assert (tmp / "h" / "HyReCo.zip").is_file()


def test_no_failure_is_swallowed():
    for p in DL.glob("*.sh"):
        assert "|| true" not in p.read_text(), p
