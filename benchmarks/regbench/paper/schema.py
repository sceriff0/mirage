"""The paper layer's view of the table schema. The implementation lives in
``benchmarks.regbench.table_schema`` so that ``score.py`` can use it without importing this
package; paper-only tables are added here."""
from benchmarks.regbench.table_schema import *  # noqa: F401,F403
from benchmarks.regbench.table_schema import (  # noqa: F401
    NA,
    TABLES,
    Col,
    Table,
    columns,
    empty_row,
    read,
    write,
)

__all__ = ["NA", "TABLES", "Col", "Table", "columns", "empty_row", "read", "write"]
