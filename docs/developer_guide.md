# Developer Guide

## Repository Layout

- Pipeline entrypoint: `main.nf`
- Subworkflows: `subworkflows/local/`
- Processes/modules: `modules/local/`
- Groovy helpers: `lib/`
- Python tools: `bin/`
- Config: `nextflow.config`, `conf/*.config`
- Schema: `nextflow_schema.json`
- Tests: `tests/`

## Style and Consistency Rules

- Keep the parameter surface synchronized across:
  - `nextflow.config`
  - `nextflow_schema.json`
  - `params.*` references in code
- Run:
  - `python3 tests/check_param_consistency.py`
- Prefer shared helpers for duplicated channel/pairwise logic (`lib/ChannelUtils.groovy`).
- Keep debug channel output gated behind `debug_channels`.

## Adding a New Module

1. Add process file under `modules/local/`.
2. Include it in the relevant subworkflow.
3. Add process resource config in `conf/modules.config` if needed.
4. Add/adjust tests (`tests/modules/` and/or subworkflow tests).
5. Update docs and schema if new parameters are introduced.

## Python Script Pattern

Use consistent script shape:

- `parse_args() -> argparse.Namespace`
- `main() -> int`
- `if __name__ == '__main__': raise SystemExit(main())`
- Shared logging via `bin/utils/logger.py`

