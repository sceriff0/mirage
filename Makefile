# MIRAGE Pipeline - Test Runner
#
# Quick reference:
#   make test              → Fast stub + python tests (~5 min)
#   make test-real         → Real container tests (~15 min, needs Docker)
#   make test-integration  → Full end-to-end (~30+ min, needs Docker)
#   make test-all          → Everything
#
# Prerequisites:
#   pip install numpy tifffile pytest pytest-cov pandas scikit-image
#   nf-test installed (https://www.nf-test.com/)
#   Docker running (for real/integration tests)

.PHONY: testdata test test-stub test-real test-integration test-python test-validation test-lint test-all clean-test help \
        arm-plan arm-run arm-plan-subset arm-rerun arm-tables sweep-tables arm-pull run-resources

# Default target
test: test-stub test-python

help:
	@echo "MIRAGE Test Targets:"
	@echo "  make test              Quick: stub + python (~5 min)"
	@echo "  make test-stub         nf-test stub-only tests"
	@echo "  make test-real         nf-test real container tests (~15 min)"
	@echo "  make test-integration  Full integration tests (~30+ min)"
	@echo "  make test-python       Python unit tests (pytest)"
	@echo "  make test-validation   Input validation tests (dry-run)"
	@echo "  make test-lint         nf-core lint"
	@echo "  make test-all          Run everything"
	@echo "  make testdata          Generate test data"
	@echo "  make clean-test        Remove test artifacts"
	@echo ""
	@echo "MIRAGE Real-Sample Benchmark (docs/benchmarks_real.md):"
	@echo "  make arm-plan          Expand arms.yaml -> arm_plan.csv + arms.csv"
	@echo "  make arm-run           Launch every arm (cluster)"
	@echo "  make arm-plan-subset   Expand only the arms a code change affects (CHANGED=tiled ... / ONLY=<regex>)"
	@echo "  make arm-rerun         Re-launch that subset into the SAME root, previous results moved aside (cluster)"
	@echo "  make arm-tables        Emit the ARM tables into _handoff/arms/"
	@echo "  make sweep-tables      Emit the SWEEP tables (needs SWEEP=<root>)"
	@echo "  make arm-pull          Copy the artifacts into ihc_method/data/"
	@echo "  make run-resources     ONE run's CPU/wall/peak-RSS profile -> ihc_method/data/run_resources/ (RUN=<outdir>)"
	@echo "    Variables: INPUT=real_input.csv ROOT=arm_results IHC=../ihc_method"
	@echo "               SWEEP=sweep_results (adds the resource sweep to the pull)"

# Generate test data (prerequisite for all test targets)
testdata:
	@echo "Generating test data..."
	python tests/testdata/generate_complete_testdata.py

# Tier 1: Fast stub tests — runs in CI on every push/PR (< 5 min)
# Matches CI's actual blocking gate (.github/workflows/ci.yml): container-free
# (--profile test, no docker) and tag-filtered (--tag stub). --profile test,docker
# pulls (and on arm64, qemu-emulates) container images even in stub mode, since
# stub only skips script: execution, not image resolution -- that combination hangs.
test-stub: testdata
	nf-test test --tag stub --profile test --verbose

# Tier 2: Real execution tests with containers — runs in CI on main/dev push (~15 min)
test-real: testdata
	nf-test test --profile test,docker --tag real --verbose

# Tier 3: Full integration tests — manual from terminal only (~30+ min)
test-integration: testdata
	nf-test test --profile test,docker --tag integration --verbose

# Registration parity — proves classic == distributed (SEPARATED default path is bit-identical, and
# the tiled path is bit-identical to VALIS's in-process tiler). Needs Docker + the patched VALIS image.
# Exits non-zero if the paths diverge (compare_classic_vs_distributed.py's PARITY GATE).
REG_DIST_IMAGE ?= bolt3x/attend_image_analysis:mirage_valis_1.0.0
test-registration-parity: testdata
	docker run --rm -v "$(PWD)":/work -w /work $(REG_DIST_IMAGE) \
		python3 tests/integration/verify_distributed_bitidentical.py
	docker run --rm -v "$(PWD)":/work -w /work $(REG_DIST_IMAGE) \
		python3 tests/integration/compare_classic_vs_distributed.py

# Python unit tests — runs in CI on every push
test-python: testdata
	pytest -v tests/ --ignore=tests/testdata --ignore=tests/modules --ignore=tests/subworkflows --ignore=tests/integration

# Validation tests (dry-run parameter validation) — runs in CI
test-validation: testdata
	bash tests/run_validation_tests.sh

# nf-core lint — runs in CI
test-lint:
	nf-core lint .

# Run everything
test-all: test-stub test-real test-integration test-python test-validation test-lint

# Cleanup test artifacts
clean-test:
	rm -rf .nf-test test_results test_results_stub test_results_full
	rm -rf work/ .nextflow.log* .nextflow/
	@echo "Test artifacts cleaned"

# =============================================================================
# Real-sample arm benchmark  (docs/benchmarks_real.md)
# =============================================================================
# The four steps are separate targets on purpose: `arm-run` is hours-to-days of
# cluster time, so it must never be a transparent dependency of a target you
# reach for to regenerate a table. `arm-tables` and `arm-pull` re-read whatever
# has finished and are safe to repeat while the sweep is still going.

INPUT ?= real_input.csv
ROOT  ?= arm_results
IHC   ?= ../ihc_method
ARMS  ?= benchmarks/configs/arms.yaml
SWEEP ?=
SWEEP_PLAN ?= $(SWEEP)_plan.csv
HANDOFF ?= benchmarks/_handoff

# An OPTIONAL landmark-TRE CSV (see load.GROUND_TRUTH_COLS). Its producer is
# benchmarks/anhir/evaluate.py (anhir_reg_eval.csv, from the public ANHIR
# landmarks) -- see benchmarks/README.md section B and benchmarks/anhir/README.md.
# make_figures requires --reg-eval: it used to sit unread in that function's
# signature, so the number reached no table and no figure at all.
#
# The default is the explicit opt-out. That is not the same as omitting it: it
# writes NO_GROUND_TRUTH.txt into the output directory, so a cost-only result
# says so in the deliverable rather than reading like a complete one. Set
# REG_EVAL=<path to an external landmark-TRE csv> if you have one.
REG_EVAL ?= none

arm-plan:
	python benchmarks/build_arm_plan.py \
	    --arms $(ARMS) --input $(INPUT) \
	    --out $(ROOT)_plan.csv --results-root $(ROOT)

arm-run: arm-plan
	benchmarks/run_arms.sh $(ROOT)_plan.csv $(INPUT) $(ROOT)

# Re-running a SUBSET after a code change (docs/benchmarks_real.md, "Re-running a
# subset after a code change"). CHANGED is a space-separated list of components
# (tiled|stare|valis|qc|preprocess|seg:<method>, see benchmarks/impact.py), ONLY a
# regex on arm/run_id; build_arm_plan.py takes the transitive closure (a QC cross
# of a re-run base re-runs, and so on). The subset plan gets ITS OWN file so
# $(ROOT)_plan.csv -- the FULL plan that arm-tables and arm-pull read -- is never
# overwritten by a subset, and arms.csv is rewritten from the full plan. arm-rerun
# launches with ARMS_REPLACE=1, which moves exactly those arms' previous results
# aside to $(ROOT)/.replaced/<timestamp>/ (never deleted). Like arm-run, it is NOT
# a prerequisite of any table target: hours-to-days of cluster time must never be
# a transparent dependency of a table you reach for.
CHANGED ?=
ONLY ?=
SUBSET_PLAN ?= $(ROOT)_plan.subset.csv

arm-plan-subset:
	@[ -n "$(CHANGED)$(ONLY)" ] || { echo "set CHANGED=<tiled|stare|valis|qc|preprocess|seg:<method> ...> and/or ONLY=<regex>"; exit 1; }
	python benchmarks/build_arm_plan.py \
	    --arms $(ARMS) --input $(INPUT) \
	    --out $(SUBSET_PLAN) --results-root $(ROOT) \
	    $(foreach c,$(CHANGED),--changed $(c)) $(if $(ONLY),--only '$(ONLY)',)

arm-rerun: arm-plan-subset
	ARMS_REPLACE=1 benchmarks/run_arms.sh $(SUBSET_PLAN) $(INPUT) $(ROOT)

arm-tables:
	python -m benchmarks.analysis.make_tables \
	    --results-root $(ROOT) --run-plan $(ROOT)_plan.csv --outdir $(HANDOFF)/arms
	python -m benchmarks.analysis.make_figures \
	    --results-root $(ROOT) --run-plan $(ROOT)_plan.csv --reg-eval $(REG_EVAL) \
	    --outdir $(HANDOFF)/arms

# The sweep is a DIFFERENT experiment that writes the same filenames. Its tables
# are what data/benchmark/ must hold, so they get their own staging dir and their
# own target -- sequencing must never decide which one a page reads.
sweep-tables:
	@[ -n "$(SWEEP)" ] || { echo "set SWEEP=<sweep results root>"; exit 1; }
	python -m benchmarks.analysis.make_tables \
	    --results-root $(SWEEP) --run-plan $(SWEEP_PLAN) --outdir $(HANDOFF)/sweep
	python -m benchmarks.analysis.make_figures \
	    --results-root $(SWEEP) --run-plan $(SWEEP_PLAN) --reg-eval $(REG_EVAL) \
	    --outdir $(HANDOFF)/sweep

arm-pull:
	benchmarks/pull_to_ihc_method.sh $(ROOT) $(IHC) --handoff $(HANDOFF) \
	    $(if $(SWEEP),--sweep $(SWEEP) --sweep-plan $(SWEEP_PLAN),)

# ONE real run's resource profile (CPU-h, wall-time, peak RSS vs input size) for
# ihc_method's analysis/run_resources.Rmd. Any finished run, arm or not:
#   make run-resources RUN=/path/to/results [TRACE=/path/to/.trace/trace.txt]
# TRACE is needed when trace_dir was left at its default (`.trace` beside the
# launch directory) and that directory is not <RUN>/../.trace.
RUN ?=
TRACE ?=
run-resources:
	@[ -n "$(RUN)" ] || { echo "set RUN=<a mirage run's --outdir>"; exit 1; }
	benchmarks/pull_run_resources.sh $(RUN) $(IHC) --handoff $(HANDOFF) \
	    $(if $(TRACE),--trace $(TRACE),)
