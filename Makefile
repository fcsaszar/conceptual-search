.PHONY: help install check dry-run mock-smoke estimate-full check-live \
        run-full run-polish run-baseline run-baseline-entrants run-universe-scoring \
        run-championship run-baseline-championship run-semantic figures robustness-tests

# Run id of the searches reported in the paper.  Set RUN_ID=... to reproduce
# the pipeline under a new id.
RUN_ID ?= 2026-08-08_deepseekv32-evalrank15-full
PYTHON ?= python3
MODEL := deepseek/deepseek-v3.2
FIRM_SET := in/firm-set-15-evalrank.txt
STEPS := 12
CACHE := out/$(RUN_ID)-call-cache.sqlite3
export PYTHONDONTWRITEBYTECODE := 1

help:
	@printf '%s\n' \
	  'Conceptual search: code and records for the paper' \
	  '' \
	  'Setup and validation (no API calls):' \
	  '  make install            Install the Python dependencies (src/requirements.txt)' \
	  '  make check              Syntax checks and deterministic unit tests' \
	  '  make dry-run            Print the search workflow and write a manifest without running it' \
	  '  make mock-smoke         Run the one-step pipeline offline with mock model responses' \
	  '  make figures            Rebuild the paper figures and tables from the preserved records' \
	  '  make robustness-tests   Unit tests of the two cross-evaluator checks' \
	  '' \
	  'Live stages, in order (paid; require OPENROUTER_API_KEY; set RUN_ID for a new run):' \
	  '  make estimate-full             Cost estimate for run-full' \
	  '  make run-full                  Global search, local search, seed-benchmark scoring, semantic scores' \
	  '  make run-polish                Twelve rounds of local search on each final-generation concept' \
	  '  make run-baseline              Local search from each of the other 14 seeds (local-only comparison)' \
	  '  make run-baseline-entrants     Collect the 14 local-only finals into a tournament entrants file' \
	  '  make run-universe-scoring      Score every searched concept against the 30-venture reference set' \
	  '  make run-championship          Final tournament: polished descendants, parents, finals, and the 30 originals' \
	  '  make run-baseline-championship Same tournament for the local-only finals' \
	  '  make run-semantic              Semantic distances of the local-only finals and the polished slate (no API calls)' \
	  '' \
	  'The README documents every stage, its outputs, and the numbers it supports.'

install:
	$(PYTHON) -m pip install -r src/requirements.txt

check:
	$(PYTHON) -c 'import ast, pathlib; [ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in list(pathlib.Path("src").glob("*.py")) + list(pathlib.Path("analysis").glob("*.py"))]'
	$(PYTHON) -m unittest discover -s src/tests -v

dry-run:
	$(PYTHON) src/run_pipeline.py --firm-set $(FIRM_SET) --model $(MODEL) --steps $(STEPS) \
	  --run-id dry-run-check --run-kind full --quality-scope all-points --dry-run

mock-smoke:
	$(PYTHON) src/run_pipeline.py --firm-set $(FIRM_SET) --model $(MODEL) --steps 1 \
	  --run-id offline-smoke --run-kind smoke --quality-scope all-points --max-concurrent-calls 20 --mock

estimate-full:
	$(PYTHON) src/estimate_cost.py --firms 15 --steps $(STEPS) --model $(MODEL) \
	  --common-quality seed-benchmark --common-quality-scope all-points

check-live:
	@test -n "$$OPENROUTER_API_KEY" || { printf '%s\n' 'ERROR: Export OPENROUTER_API_KEY before starting a paid run.'; exit 2; }

run-full: check-live
	$(PYTHON) src/run_pipeline.py --firm-set $(FIRM_SET) --model $(MODEL) --steps $(STEPS) \
	  --quality-calibration seed-benchmark --quality-scope all-points \
	  --run-id "$(RUN_ID)" --run-kind full --max-concurrent-calls 40

run-polish: check-live
	$(PYTHON) src/polish_population.py --run-id "$(RUN_ID)" --model $(MODEL) --steps $(STEPS) --per-run-concurrency 8

run-baseline: check-live
	$(PYTHON) analysis/run_baseline15.py --run-id "$(RUN_ID)" --firm-set $(FIRM_SET) --model $(MODEL) --steps $(STEPS)

run-baseline-entrants:
	$(PYTHON) analysis/collect_baseline15.py --run-id "$(RUN_ID)" --phase entrants

# Every unique concept from the focal local and global searches against all 30
# reference ventures, both presentation orders.  The seed-benchmark validation
# expects the 30-venture firm set here.
run-universe-scoring: check-live
	AI_ENTREP_EXPECTED_FIRM_COUNT=30 AI_ENTREP_CALL_CACHE=$(CACHE) AI_ENTREP_MAX_CONCURRENT_CALLS=40 \
	$(PYTHON) src/common_quality.py \
	  --local-csv out/$(RUN_ID)-local-search.csv --global-csv out/$(RUN_ID)-global-search.csv \
	  --firm-set in/firm-set-30.txt --model $(MODEL) --run-prefix "$(RUN_ID)-universe" \
	  --input-file in/projects.csv --output-dir out --scope all-points

run-championship: check-live
	AI_ENTREP_CALL_CACHE=$(CACHE) AI_ENTREP_MAX_CONCURRENT_CALLS=40 $(PYTHON) src/championship.py \
	  --input-file in/projects.csv --commit-round out/$(RUN_ID)-commit-round.csv \
	  --local-candidates out/$(RUN_ID)-local-candidates.csv --model $(MODEL) \
	  --run-prefix "$(RUN_ID)" --output-dir out --entrants-csv out/$(RUN_ID)-polish-entrants.csv

run-baseline-championship: check-live
	AI_ENTREP_CALL_CACHE=$(CACHE) AI_ENTREP_MAX_CONCURRENT_CALLS=40 $(PYTHON) src/championship.py \
	  --input-file in/projects.csv --commit-round out/$(RUN_ID)-commit-round.csv \
	  --local-candidates out/$(RUN_ID)-local-candidates.csv --model $(MODEL) \
	  --run-prefix "$(RUN_ID)-baseline15" --output-dir out --entrants-csv out/$(RUN_ID)-baseline15-entrants.csv

run-semantic:
	$(PYTHON) src/semantic_scores.py --input-file in/projects.csv --firm-set $(FIRM_SET) \
	  $(foreach f,$(wildcard out/$(RUN_ID)-baseline15-*-local-candidates.csv),--candidate-csv $(f)) \
	  --output out/$(RUN_ID)-baseline15-semantic-scores.csv
	$(PYTHON) src/semantic_scores.py --input-file in/projects.csv --firm-set $(FIRM_SET) \
	  --candidate-csv out/$(RUN_ID)-polish-entrants.csv \
	  --output out/$(RUN_ID)-polish-semantic-scores.csv

# Figures and tables from the preserved records (no API calls).
figures:
	$(PYTHON) analysis/compute_universe_wr.py --run-id "$(RUN_ID)"
	$(PYTHON) analysis/build_local_figs.py --run-id "$(RUN_ID)" --seed-id clockchain --firm-set $(FIRM_SET)
	$(PYTHON) analysis/build_genealogy_tree.py --run-id "$(RUN_ID)" --feature-slot g12-s04
	$(PYTHON) analysis/build_search_map.py --run-id "$(RUN_ID)" --seed-id clockchain --feature-slot g12-s04
	$(PYTHON) analysis/build_polish_figure.py --run-id "$(RUN_ID)" --firm-set $(FIRM_SET)
	$(PYTHON) analysis/build_universe_table.py
	$(PYTHON) analysis/collect_baseline15.py --run-id "$(RUN_ID)" --phase summary

robustness-tests:
	cd robustness1-gemma3-4b && $(PYTHON) -m unittest -v test_robustness.py
	cd robustness2-qwen2.5-72b && $(PYTHON) -m unittest -v test_robustness.py
