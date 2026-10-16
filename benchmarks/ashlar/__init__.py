"""ASHLAR comparator drivers, harness-owned.

ASHLAR is the field-standard external cyclic-IF registrar and the benchmark's only
EXTERNAL baseline. It is deliberately NOT a Mirage registration backend -- dev removed
that at :fire: 6a54479 and guards it with tests/test_ashlar_backend_removed.py -- so a
ranking that includes it has to drive it itself. That is what this package is.

``retile`` cuts a slide into the tile grid ASHLAR reads, and ``original`` runs the
original ASHLAR on it; benchmarks/regbench imports both. The earlier solve step (placements
rewritten into STARE's M0 + mesh manifest) and its arm driver are removed.

Scored any other way it would produce numbers in a different metric family sharing no
column with the arm table -- exactly what the earlier out-of-band ashlar harness did, and
what the deleted synthetic ground-truth rung did after it.
"""
