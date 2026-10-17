"""The paper layer: reads ``<out>/tables`` and the result records, writes ``<out>/paper``.
It is never imported by a method adapter or by ``score.py``; ``score.py`` uses
``benchmarks.regbench.table_schema`` directly, and ``paper.schema`` re-exports it."""
