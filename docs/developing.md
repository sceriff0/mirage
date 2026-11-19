# Developing

This page is for someone changing MIRAGE, not running it. It covers the
conventions the test suite enforces, the specific ways that suite can look green
while proving nothing, and the checklist for adding a process.

Everything here is checked by a test. Where a rule names its guard, that guard is
the authority — read it before arguing with the prose.

## Layout

```
main.nf                       entry point; calls the MIRAGE workflow
workflows/mirage.nf           step routing and QC aggregation
subworkflows/local/           one subworkflow per stage; adapters/ holds the
                              interchangeable registration backends
modules/local/                one process per file, UPPER_SNAKE_CASE
conf/                         base, modules (publishDir + ext.args), test,
                              and a site profile template
lib/                          Groovy classes, all methods static
bin/                          the Python the processes run
tests/                        pytest guards, nf-test suites, shell checks
```

## Conventions

**Parameter defaults live in `nextflow.config`, once.** Never re-declare one
elsewhere as `params.x ?: 'value'` — a second declaration silently wins or
diverges. Guarded by `tests/test_no_duplicate_param_defaults.py`, which scans
`modules/local/`, `conf/modules.config`, `workflows/` and `subworkflows/`.

**Published paths have one owner: `lib/Layout.groovy`.** Never hand-write a
published path or a `csv/*.csv` filename. `modules/local/*.nf` know nothing about
publish layout, and `tests/test_layout.py` keeps it that way.

**Every process has exactly one resource owner** — either a `label` or a
`withName:` block in `conf/modules.config` setting all three of `cpus`, `memory`
and `time`. Never both (a `withName:` block wins, making the label inert and
misleading) and never neither (the process silently inherits `conf/base.config`'s
small defaults and OOMs on a retry ramp it was never sized for).
`tests/test_resource_label_coverage.py` fails on both mistakes; the current split
is tabulated in [Resources](resources.md).

**Failure policy is a small named set, inlined verbatim at each site.** There are
four bodies — retry-then-fail, retry-exit1-then-fail, retry-then-drop, fail-fast —
and a new `withName:` block picks one and writes it out. It cannot be shared from a
helper (see the next rule), so naming *is* the abstraction here.
`tests/test_error_strategy_policy.py` rejects anything else, including the one
combination that is deliberately absent: select on a specific exit cause and then
*drop* the task, which would silently discard an ordinary `exit 1`.

**`conf/*.config` cannot see `lib/*.groovy`, and fails silently when it tries.**
Nextflow parses config in a separate pass, so a class name inside a config closure
does not fail to compile — it resolves against `ConfigObject` and surfaces as a
misleading `MissingMethodException` only when the closure runs. The same applies to
`log`: an `errorStrategy` closure calling `log.warn` aborts the run on Nextflow 25
(which the manifest accepts) while working on 26. Anything that needs a
`lib/` class, or needs to say something to the operator, belongs in a module's
`script:` block or in `main.nf`'s `onComplete`.

There is also **no legal way to share a helper between two `withName:` blocks**
under Nextflow 26's strict parser. A top-level `def helper(...)` in a config file
fails to parse; a top-level closure assignment is rejected as a dynamic option
outside process scope. Inline the logic at each call site.

**Containers are tagged by content, never `:latest`.** Every process declares one.

**Tool arguments go in `conf/modules.config` via `ext.args`**, not hardcoded in a
process script.

**A boolean parameter cannot be passed on the command line.** Nextflow 26 hands
every `--param` to the schema validator as a String, so both command-line forms of
a boolean — with an explicit value and bare — are rejected as "[string] but should
be [boolean]". Use a `-params-file` (there are ready-made ones under `params/`) or
a profile. `tests/test_no_cli_boolean_params_in_docs.py` enforces this across the
repository *including inside comments*, so document the trap without writing the
broken form, as this paragraph does.

**Validation is two layers and both run before any process starts.** Single-value
types, enums and ranges come from `nextflow_schema.json` through nf-schema's
`validateParameters()`, called once in `workflows/mirage.nf`; cross-parameter and
filesystem rules live in `lib/ParamUtils.groovy`. The dry-run mode exercises
exactly that surface and nothing else.

## Verification reality — read this before trusting a green run

Each of these makes a green run prove less than it seems to.

**1. `-stub` never evaluates a `script:` block.** A stub run exercises channel
wiring and the `stub:` block only, so any change to a *rendered command* is
invisible to it. If you change what a process is invoked with, add an nf-test that
asserts the command string. Those tests are tagged `stub` so CI collects them, but
they carry no `-stub` option — stub mode would substitute away the very command
under assertion. `tests/modules/generate_qc_report.nf.test` and
`tests/modules/seg_qc_geojson.nf.test` are the precedents.

**2. CI's blocking nf-test gate is `nf-test test --tag stub`.** A file without that
tag never runs there. Tag new files or they are decorative.

**3. An `errorStrategy` that ignores a failure drops the task without failing the
run.** The default-on QC selector fails the run. Two processes keep a deliberate
drop, both opt-in, and when either drops, the completion handler announces the count
rather than leaving it to be inferred from a missing column. Still: check published
artefacts, not just the exit code.

**4. `-resume` can silently reuse nothing, or re-run everything, and stay green.**
Two hashing rules bite. A process `script:` block that references `params` hashes
the **whole map**, so any unrelated parameter change re-runs that task and
everything downstream — pass scalars, never the map. And `groupTuple()` emits in
arrival order while Nextflow hashes list inputs positionally, so an identical rerun
can miss and cascade; the fix is to transpose and sort explicitly, never
`groupTuple(sort:)`, which orders each grouped list independently and silently
re-pairs metadata with the wrong file. Separately, `collectFile()` is an operator
rather than a task and rewrites to a fresh temporary directory every run, so a
process consuming its output can never cache — the two report processes that do are
declared uncacheable for exactly that reason, and seeing only those two re-run on
an otherwise fully-cached resume is correct. `tests/test_resume_determinism.py`
guards the mechanism statically, because a cache miss of this kind is intermittent
and a run-twice check can pass by luck; `tests/resume_check.sh` is the behavioural
counterpart.

**5. A guard that parses Nextflow source must use `tests/nfmodel`**, never a
private regex: a naive comment-stripping regex is fooled by `stageAs: 'ref/*'`,
which opens a fake block comment and hides the code after it. Pick the right view,
too: the comments-and-strings-blanked view locates code and is right for
identifier scans; the strings-preserved view is the one to use whenever the thing
under assertion lives inside a quoted literal, which includes every published path,
every `ext.args` flag and every failure-policy body.

**6. A test can only run where its fixture exists.** `tests/testdata/` is
gitignored, so a fixture is available in CI only if
`tests/testdata/generate_complete_testdata.py` writes it. New fixtures go in that
generator *and* in `.gitignore`'s per-file list, and
`tests/test_fixtures_have_a_producer.py` checks both that they exist and that they
open as what their suffix claims (a zero-byte fixture exists but is not a TIFF).

Related: `tests/lib_probe.nf` is the only unit-test surface `lib/*.groovy` has,
because nf-test's assertion context cannot see `lib/` classes. It must parse under
Nextflow 26 as well as 25, or every assertion in it is silently skipped in one CI
leg.

**7. A text-matching guard is satisfied by a comment unless you make it
comment-blind.** A comment naming a command satisfies a naive check even when the
real command has been replaced, and a commented-out gate still contains the string
the guard looks for. The rule: a
guard asking *"does this actually run?"* reads the comment-stripped view
(`tests/ci_actions.py` has the one quote-aware definition — never a private
`split("#")`, which truncates shell parameter expansions). The raw view is right
only when the file's literal bytes are the subject. Negative rules — assertions
that something must **not** appear — want the opposite treatment: there a trailing
comment should make the guard red, i.e. noisy rather than blind.

**And: real (non-stub) nf-test does not run reliably on arm64 under emulation**, and
`-profile test,docker` hangs pulling images there. The fast local loop is the
container-less stub:

```bash
nextflow -q run . -profile test -stub -w /tmp/_nfw --outdir /tmp/_stub
```

**Every documented command is checked.** `tests/test_documented_commands_are_runnable.py`
checks that every `nextflow run` command in the README and the docs carries
`--outdir` and the sizing pair (`-c site.config`, a pinning profile, or a preset that
carries both) — a static token check, run in the Python suite.
`tests/documented_commands_launch.sh` goes further: it substitutes each command's
placeholders for this repo's own fixtures and runs
`nextflow -stub -params-file params/dry_run.json` against every one, so an
unrecognised flag or a schema rejection fails CI too. A command that is illustrative
only carries a `# not-runnable: <reason>` comment on the line above it.

## Adding a process

1. Create `modules/local/my_process.nf` from the standard template: `tag`,
   `container`, meta-carrying input and output, a `versions.yml` emit, and a
   `stub:` block. One process per file, UPPER_SNAKE_CASE — checked by
   `tests/test_module_conventions.py`.
2. Give it **exactly one resource owner**: a `label`, or a `withName:` block in
   `conf/modules.config` setting all of `cpus`/`memory`/`time`. Never both, never
   neither.
3. Add its `publishDir` and `ext.args` in `conf/modules.config`. Remember that file
   cannot reference `lib/` classes.
4. Include it in the relevant subworkflow.
5. Add an nf-test under `tests/modules/` and **tag it `stub`**, or the blocking
   gate never runs it. If the rendered command carries anything tunable, assert the
   command string.
6. If the process calls a script in `bin/` **by name**, that script must be tracked
   executable, or it fails at runtime with a permission error: Nextflow stages
   `bin/` onto `PATH` and execs it directly. Use `git update-index --chmod=+x` and
   confirm with `git ls-files -s` that the mode is `100755` — a local `chmod` alone
   does not reach a fresh checkout. Import-only libraries under `bin/utils/` stay
   non-executable, and each needs a real importer
   (`tests/test_no_dead_bin_modules.py`) — a module nothing imports is deleted, not
   kept just in case.

## Working in a git worktree

- `tests/testdata/` is gitignored per worktree. Run
  `python tests/testdata/generate_complete_testdata.py` in a fresh worktree
  *before* believing any test failure; absent or stale fixtures produce failures
  that look exactly like code defects.
- **Never run `nf-test` in two worktrees at once.** They share `.nf-test/` and
  produce phantom failures.
- Pass `--basetemp=<unique dir>` to every pytest run when more than one process
  shares the virtualenv; two runs fighting over pytest's shared basetemp produce
  non-reproducible collection counts.

## Building the docs locally

The docs are a MkDocs Material site. Build and live-preview them:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open <http://127.0.0.1:8000> — the site rebuilds on every save. CI builds it with
`mkdocs build --strict`, so a broken link or anchor, or a page missing from the `nav`
of `mkdocs.yml`, fails the build.
