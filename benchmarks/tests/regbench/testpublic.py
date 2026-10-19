"""Two TEST-ONLY public datasets, for the tests of the public-dataset machinery.

``regbench.PUBLIC`` holds the real public datasets; the machinery (the own tables of a
public dataset, its run list, the ``tissue`` scoring kind, the launcher's ``--datasets``)
has to be tested on a dataset of each kind it handles, and the user ruled on 2026-10-18
that the only ``tissue`` set and a ``nuclei`` set are removed. These stand in:

``nucset``     a ``nuclei`` set, fluorescence, regime ``cyclic_if`` (as MCMICRO is)
``tissueset``  a ``tissue`` set, brightfield, regime ``bf_same`` (the one brightfield regime
               since ``bf_consecutive`` was removed, 2026-10-19)

``register`` adds them to ``regbench.PUBLIC`` and ``regimes.DATASET_REGIME`` (and the
derived ``variants.NOT_ON`` and ``cli.ALL_DATASETS`` once those are imported), through
``setitem``/``setattr`` callbacks so a fixture can undo it. A launcher run is another
process: ``sitecustomize.py`` below, put on its ``PYTHONPATH``, registers them there, and
``patched_launcher`` copies the slurm folder with the datasets added to ``PUBLIC_DATASETS``
and ``PUBLIC_KINDS``. Nothing here reaches a result: it is imported by tests only.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

NUCLEI = "nucset"
TISSUE = "tissueset"
SPECS = {
    NUCLEI: {"kind": "nuclei", "modality": "fluorescence"},
    TISSUE: {"kind": "tissue", "modality": "brightfield"},
}
REGIMES = {NUCLEI: "cyclic_if", TISSUE: "bf_same"}

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"

_SITECUSTOMIZE = """\
import sys

sys.path.insert(0, {repo!r})
from benchmarks.tests.regbench import testpublic

testpublic.register()
"""


def register(setitem=None, setattr_=None):
    """Add the two datasets. With no callbacks the change is permanent (a launcher
    subprocess); a fixture passes ``monkeypatch.setitem`` / ``monkeypatch.setattr``."""
    from benchmarks import regbench
    from benchmarks.regbench import regimes

    setitem = setitem or (lambda d, k, v: d.__setitem__(k, v))
    setattr_ = setattr_ or setattr
    for name, spec in SPECS.items():
        setitem(regbench.PUBLIC, name, dict(spec))
        setitem(regimes.DATASET_REGIME, name, REGIMES[name])
    variants = sys.modules.get("benchmarks.regbench.variants")
    if variants is not None:
        setitem(variants.NOT_ON, TISSUE, variants._FLUO_ON_RGB)
    cli = sys.modules.get("benchmarks.regbench.cli")
    if cli is not None:
        setattr_(cli, "ALL_DATASETS", cli.ALL_DATASETS + tuple(SPECS))


def patched_launcher(tmp):
    """``(script, env)``: a copy of the slurm folder in ``tmp`` whose launcher lists the
    two datasets (``PUBLIC_DATASETS`` and ``PUBLIC_KINDS``), and the environment that
    makes a python started by it know them too."""
    here = Path(tmp) / "slurm"
    shutil.copytree(SLURM, here)
    script = here / "submit_cohort.sh"
    text = script.read_text()
    kinds = " ".join(f"{n}={s['kind']}" for n, s in SPECS.items())
    names = " ".join(SPECS)
    old_names = 'PUBLIC_DATASETS="'
    old_kinds = 'PUBLIC_KINDS="'
    assert text.count(old_names) == 1 and text.count(old_kinds) == 1
    text = text.replace(old_names, f'PUBLIC_DATASETS="{names} ')
    text = text.replace(old_kinds, f'PUBLIC_KINDS="{kinds} ')
    script.write_text(text)
    site = Path(tmp) / "sitecustomize_dir"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE.format(repo=str(REPO)))
    return script, {
        "REGBENCH_REPO": str(REPO),
        "PYTHONPATH": os.pathsep.join([str(site), *filter(None, [os.environ.get("PYTHONPATH")])]),
    }
