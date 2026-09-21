# Conceptual Search

Code, prompts, inputs, and complete search records for the conceptual-search
exercise reported in

> Csaszar, F. A. (2026). Conceptual search: A generative view of
> entrepreneurial imagination. *Strategic Entrepreneurship Journal*,
> forthcoming.
>
> Preprint: <https://ssrn.com/abstract=7496559>

*Conceptual search* treats entrepreneurial imagination as search over venture
concepts written in natural language.  The software in this repository builds
and runs such a search with a large language model: it generates venture
concepts, evaluates them in pairwise tournaments, refines them (local search),
recombines them (global search), checks every generated text for invented
evidence, and measures every concept against a fixed reference set of thirty
real crowdfunding campaigns.  The repository also preserves the complete
record of the searches reported in the paper, so every figure, table, and
number in the paper's empirical section can be rebuilt from the records
without a single model call.

Author: Felipe A. Csaszar, Ross School of Business, University of Michigan
(<https://csaszar.info>).

## Citation

This repository contains the code and complete search records for:

Csaszar, F. A. (2026). Conceptual search: A generative view of entrepreneurial
imagination. *Strategic Entrepreneurship Journal*, forthcoming.
A preprint is available at <https://ssrn.com/abstract=7496559>.

If you use this code or the search records in your research, please cite the paper:

```bibtex
@article{csaszar2026conceptual,
  author  = {Csaszar, Felipe A.},
  title   = {Conceptual Search: A Generative View of Entrepreneurial Imagination},
  journal = {Strategic Entrepreneurship Journal},
  year    = {2026},
  note    = {Forthcoming}
}
```

A machine-readable citation is in `CITATION.cff`.

## Contents

| Path | What it holds |
|---|---|
| `src/` | The search pipeline: model client, prompts, local search, global search, fidelity checks, semantic distance, tournaments, orchestration, and unit tests (`src/tests/`). |
| `in/` | Inputs: the 30 anonymized campaign descriptions (`projects.csv`), the seed set (`firm-set-15-evalrank.txt`), the full reference set (`firm-set-30.txt`), and the final-generation concepts that entered polish (`polish-seeds.csv`). |
| `out/` | The complete record of the reported run (run id `2026-08-08_deepseekv32-evalrank15-full`): every candidate, prompt, tournament match, fidelity decision, lineage, log, and the run manifest.  About 265 MB of CSV and log files. |
| `analysis/` | Scripts that turn the records into the paper's figures, tables, and summary numbers, plus three small derived data files. |
| `figs/` | The four figures (PDF and PNG) and two tables (LaTeX) of the paper's empirical section, as rebuilt from `out/`. |
| `robustness1-gemma3-4b/`, `robustness2-qwen2.5-72b/` | The two cross-evaluator robustness checks: code, frozen inputs, and every provider response. |
| `Makefile` | One target per stage; `make help` lists them. |

## The method in brief

Each venture concept is a Markdown text of about 500 words with six components:
problem, customer, solution, delivery model, revenue logic, and distinctiveness.
An LLM evaluator (DeepSeek V3.2) judges which of two concepts is more
promising; each pair is judged in both presentation orders.

- **Local search** (`src/local_search.py`) is hill climbing.  Each round an
  advisor prompt proposes five small changes to the incumbent, a rewriter
  produces one full variant per change, a fidelity auditor checks each variant,
  and a six-concept double round robin picks the next incumbent.  Twelve rounds.
- **Global search** (`src/global_search.py`) is a genetic algorithm over a
  population of fifteen concepts.  Each generation runs a 210-judgment double
  round robin, ranks concepts by an equal-weight average of tournament
  percentile and semantic-distance percentile, and builds the next generation
  from five elites (at most two per founding seed's family), five
  single-component mutations, and five component-level crossovers.  Twelve
  generations.  After the generations, a commit round re-verifies every concept
  that ever held an elite slot against the seeds and a seed tournament places
  the committed champion among the seeds; the paper uses the committed champion
  only as an entrant in the final tournament.
- **Fidelity checks** (`FidelityAuditor` in `src/agents_common.py`,
  `filter_crossover_audit` in `src/global_search_agents.py`) let generated
  texts propose new features, tests, or partners but reject texts that claim
  accomplishments their source does not support.  A rejected text is replaced by
  an unchanged copy of its parent; an audit that cannot be completed stops the
  run instead of counting as a rejection.
- **Polish** (`src/polish_population.py`) applies twelve rounds of local
  search to each concept of global search's final generation.
- **Local-only comparison** (`analysis/run_baseline15.py`) runs the same
  twelve-round local search from each of the fourteen other seeds.
- **Measurement.**  *Performance* is a concept's win rate against the thirty
  reference ventures (an original faces the other twenty-nine); it comes from
  the final tournaments (`src/championship.py`, restricted to matches against
  originals by `analysis/compute_universe_wr.py`) and, for every other searched
  concept, from bipartite scoring against all thirty (`src/common_quality.py`).
  *Semantic distance* is the mean cosine distance, over the six components,
  between a concept and its nearest seed, using BAAI's BGE-M3 embedding model at
  a pinned revision (`src/semantic_novelty.py`, `src/semantic_scores.py`).
- **Cross-evaluator robustness** re-scores the frozen end-of-search concepts
  and the thirty originals with Gemma 3 4B and Qwen 2.5 72B.

### What the search produced

Each image below is rebuilt from the records by `make figures`.

<img src="figs/local-trajectory.png" alt="Performance of the local-search incumbent, round by round" width="460">

*Twelve rounds of local search lift the seed concept `clockchain` from a 47
percent win rate against the reference set to 88 percent.  The ticks at the
right edge are the thirty reference ventures.*

<img src="figs/genealogy.png" alt="Ancestry of the best concept in the final generation" width="640">

*The strongest concept in global search's final generation descends from four
different seeds through eight crossovers and eleven single-component mutations.*

<img src="figs/search-map.png" alt="Every generated concept by semantic distance and performance" width="680">

*Local search climbs within a narrow band around its seed.  Global search
occupies a region several times wider, powered by crossovers that join distant
parents.*

<img src="figs/polish.png" alt="The final generation before and after local polish" width="640">

*Twelve rounds of polish on each final-generation concept.  Every polished
descendant outranks every seed, and the best outranks 24 of the 30 originals.*

Appendix B of the paper documents the procedures and parameters; Appendix C
reproduces the prompts.  The complete prompts are in
`src/local_search_prompts.py` and `src/global_search_prompts.py`, and the run
manifest records the exact prompt text that was sent.

## The reported run

| Item | Value |
|---|---|
| Run id | `2026-08-08_deepseekv32-evalrank15-full` (August 8, 2026) |
| Model | `deepseek/deepseek-v3.2` through OpenRouter, temperature 0.5 (fidelity auditor 0.0), reasoning disabled, 2,000 response tokens, 120 s timeout, five retries with exponential backoff |
| Reference set | 30 U.S. Kickstarter Technology campaigns launched in 2025, anonymized (`in/projects.csv`) |
| Seeds | The bottom 15 of the reference set under the evaluator's own double round robin (`analysis/evalrank-30.csv`; ranks 15 and 16 tied and were broken head-to-head): clockchain, hply, angelry, lockguard, sale-finder, genaix, bugout-battery, rent-a-bee, hooper, remotion-ai, viaia, lumivisor, voyagx, kebo, my-story |
| Local-search seed | `clockchain`, the highest-ranked seed |
| Global search | population 15 = 5 elites + 5 mutants + 5 crossovers; parent pool = top 10 selection ranks; first parent drawn with probability 1/rank; second parent = best-ranked eligible concept from a different family; elite lineage cap 2; selection weights 0.5 tournament / 0.5 distance; random seed 42; 12 generations |
| Semantic distance | BGE-M3, revision `5617a9f61b028005a4858fdac845db406aefb181`, CPU inference |
| Scale | 36,768 model invocations and about $18.6 in provider charges across the stages the paper reports, about three hours of wall-clock time with up to 40 calls in flight |

### Record files

All files in `out/` share the run-id prefix.  `<mode>` is `local` or `global`;
polish runs use the prefix `<run>-polish-s00` to `-s14` and the local-only
comparison runs use `<run>-baseline15-<seed>`.

| File | Contents |
|---|---|
| `<run>-manifest.json` | Settings, prompts, input hashes, stage commands, timing, and call-ledger totals of the main pipeline |
| `<run>-<mode>-candidates.csv` | One row per concept per step: text, parent(s), operator, requested change, tournament rank and win rate, selection scores, fidelity decision, the full prompt, and founding-seed lineage |
| `<run>-<mode>-matches.csv` | Every pairwise judgment with the evaluator's written analysis |
| `<run>-<mode>-fidelity-audits.csv` | Every fidelity check: model verdict, deterministic decision basis, flagged claims, generated and evaluated text |
| `<run>-<mode>-search.csv`, `-search.log`, `-search-narrative.txt` | Per-step wide table, run log with call counts, and a model-written narrative of the search |
| `<run>-commit-round.csv` | The global search's commit-by-verification round |
| `<run>-universe-common-quality.csv`, `-universe-common-quality-matches.csv` | Every searched concept (171 distinct texts) scored against the 30 reference ventures, both orders |
| `<run>-semantic-scores.csv` | Semantic distances of every local and global candidate, with per-component nearest seeds |
| `<run>-polish-map.csv`, `<run>-polish-entrants.csv` | Final-generation slots and their polished descendants |
| `<run>-championship.csv`, `-championship-matches.csv`, `-championship.log` | The final 64-entrant tournament: 30 originals, 15 polished descendants, 14 distinct pre-polish parents, the focal local final, the global champion, the best hybrid, and two runners-up |
| `<run>-baseline15-entrants.csv`, `-baseline15-championship*.csv`, `-baseline15-semantic-scores.csv` | The local-only comparison: its 14 finals, their 49-entrant tournament, and the distances of all its candidates |
| `<run>-polish-semantic-scores.csv` | Semantic distances of the 15 polished descendants (computed after the run from the preserved texts with the pinned embedding model; no model calls).  Their pre-polish parents are in `<run>-semantic-scores.csv`. |

`analysis/evalrank-30.csv` is the evaluator's ranking of the 30 originals
(their win rates against the other 29, identical to the originals' entries in
`analysis/universe-champ-wr.json`).  `analysis/universe-champ-wr.json` and
`analysis/baseline15-wr.json` hold the reference-set win rates of every
tournament entrant; both are regenerated by `make figures`.

In file and column names, `universe` denotes the 30-venture reference set,
`archive` and `benchmark` denote the 15 seeds, `novelty` denotes semantic
distance, and `quality` denotes tournament performance.

## Rebuilding the figures and tables (no model calls)

Requirements: Python 3.11 or later (3.13 was used), `pandas`, `matplotlib`, and
Graphviz (`dot`) for the genealogy layout.  The figures use the Arial font when
it is installed and fall back otherwise.

```bash
make figures
```

| Output | Paper element |
|---|---|
| `figs/local-trajectory.pdf` | Performance of the local-search incumbent, round by round |
| `figs/revisions.tex` | The accepted revision behind each round (the manuscript lightly edited three rows' wording) |
| `figs/genealogy.pdf` | Ancestry of the best concept in global search's final generation |
| `figs/search-map.pdf` | Every generated concept by semantic distance and performance |
| `figs/polish.pdf`, `figs/polish-facts.json` | The final generation before and after polish, and the polish statistics |
| `figs/reference-set.tex` | The appendix table of the thirty reference ventures |
| printed summary | Local-only comparison: performance and semantic-distance medians |

The rebuilt PNG figures, LaTeX tables, and derived JSON files are
byte-identical to the versions in this repository.  The PDFs are identical in
content but carry an embedded creation timestamp, so they differ as files.
Running `make robustness-tests` rewrites the `created_at_utc` field of the two
`inputs/sample-manifest.json` files, which leaves those two files modified in
an otherwise clean working tree.

### Where the paper's numbers come from

| Statement in the paper | Source in this repository |
|---|---|
| Reference-set ranking; seeds average 28 % against 72 % for the withheld half | `analysis/evalrank-30.csv`; `figs/polish-facts.json` (`archive_mean_wr`, `held_mean_wr`) |
| clockchain rises from 47 % to 88 % in twelve rounds and would place third | `figs/revisions.tex`; `analysis/universe-champ-wr.json` (`local-final (L12.4)`) |
| Fidelity checks rejected 171 of 1,080 texts (4/60 rewrites, 16/60 mutations, 2/60 crossovers, 149/900 polish rewrites) | `passed` column of `<run>-local-fidelity-audits.csv`, `<run>-global-fidelity-audits.csv`, `<run>-polish-s??-local-fidelity-audits.csv` |
| Best of the final generation: four seeds, eight crossovers, eleven mutations; 40 % before polish, 73 % after, ahead of 24 of 30 originals | `analysis/build_genealogy_tree.py` output; `<run>-global-candidates.csv`; `analysis/universe-champ-wr.json` (`g12-s04-parent`, `g12-s04-polished`) |
| A clockchain mutant reached 78 % by generation 4 and was gone by generation 10 | `<run>-global-candidates.csv` joined to `<run>-universe-common-quality.csv` by text |
| All fifteen polished concepts improved; median 63 % | `figs/polish-facts.json` (fourteen distinct parents; the fifteenth slot shares its parent text with another slot) |
| Local-only search: median 68 %, best 90 %; median distance 0.10, maximum 0.15 | `analysis/baseline15-wr.json`; `<run>-baseline15-semantic-scores.csv`; printed by `make figures` |
| Semantic distance of the recombined slate | `<run>-semantic-scores.csv` (final generation before polish) and `<run>-polish-semantic-scores.csv` (after polish); printed by `make figures` |
| Model invocations and costs per stage | `# of calls`, `Evaluator calls`, and `Calls:` lines in the `.log` files; token totals in `<run>-manifest.json` |
| Cross-evaluator results | `robustness*/outputs/*/report.md`, `results.json`, and `robustness2-qwen2.5-72b/outputs/*/cross-evaluator-comparison.md` |
| Appendix table of `clockchain` before and after local search | Hand-quoted passages.  The seed text is the `clockchain` row of `in/projects.csv`; the round-12 text is the last `selected` row of `<run>-local-candidates.csv`. |
| Appendix table of recombined material | Hand-quoted passages.  The concept is `g12-s04-parent` in `<run>-polish-entrants.csv`; its ancestors and their texts are in `<run>-global-candidates.csv`. |

## Re-running the searches (paid)

Live runs call OpenRouter and cost money.  The reported run's stages made
36,768 invocations for about $18.6; a fresh run of the same design costs about
the same and takes several hours.  Model responses are stochastic and provider
deployments change, so a new run produces different concepts and numbers.

Setup:

```bash
python3 -m pip install -r src/requirements.txt   # openai, tenacity, torch (CPU), sentence-transformers
export OPENROUTER_API_KEY=...                     # read at call time; never written to disk
make check                                        # syntax checks and 44 deterministic unit tests
make dry-run                                      # prints the workflow and writes a manifest
make mock-smoke                                   # one-step pipeline with offline mock responses
```

The first live stage downloads the BGE-M3 weights (about 2.3 GB) from Hugging
Face at the pinned revision.

Stages, in order (set `RUN_ID=<new id>` for a new run; the defaults reproduce
the reported run's file names):

| Command | What it does | Reported run |
|---|---|---|
| `make estimate-full` | Cost estimate for `run-full` | |
| `make run-full` | Global search (12 generations, commit round, seed tournament), local search from clockchain, scoring of every concept against the 15 seeds, semantic distances | 4,836 + 501 invocations; the seed-benchmark scoring adds 5,326 |
| `make run-polish` | Extracts the final generation to `in/polish-seeds.csv` and runs 15 parallel local searches | 7,098 |
| `make run-baseline` | 14 local searches from the other seeds | 14 × 504 minus duplicates plus retries |
| `make run-baseline-entrants` | Collects the 14 local-only finals into a tournament entrants file (no calls) | |
| `make run-universe-scoring` | Scores every concept of the focal searches against all 30 reference ventures | 10,712 |
| `make run-championship` | The 64-entrant final tournament | 4,226 |
| `make run-baseline-championship` | The 49-entrant local-only tournament | included in 9,395 |
| `make run-semantic` | Semantic distances of the local-only candidates and the polished slate (no calls) | |
| `make figures` | Figures, tables, and summaries | |

Every live response is cached in a per-run SQLite ledger
(`out/<run>-call-cache.sqlite3`, listed in `.gitignore`), keyed by model,
prompt, role, and settings.  Rerunning a command with the same run id resumes
from the ledger without paying for completed calls.  Stages that reuse the main
ledger (reference-set scoring and the tournaments) replay judgments already
made.  The reported run's ledger had been warm-started from an earlier run so
that all original-versus-original judgments were replayed identically; the
`actual_usage` block of its manifest therefore includes those inherited
entries.

Environment variables read by the code: `OPENROUTER_API_KEY`,
`AI_ENTREP_CALL_CACHE` (ledger path), `AI_ENTREP_MAX_CONCURRENT_CALLS`,
`AI_ENTREP_SEED_PROJECT_ID` (local-search seed), `AI_ENTREP_EXPECTED_FIRM_COUNT`
(15, or 30 for reference-set scoring), `AI_ENTREP_ELITE_LINEAGE_CAP`,
`AI_ENTREP_QUALITY_WEIGHT` and `AI_ENTREP_NOVELTY_WEIGHT` (selection weights),
`AI_ENTREP_EMBEDDING_DEVICE`, and `AI_ENTREP_MOCK_LLM` (offline tests).

## Cross-evaluator robustness checks

`robustness1-gemma3-4b/` and `robustness2-qwen2.5-72b/` re-score the thirty
originals and the 44 distinct end-of-search texts (final generation, polished
descendants, local-only finals) with Gemma 3 4B and Qwen 2.5 72B, using the
paper's pairwise prompt in both presentation orders (3,510 judgments each).
Each folder has its own README with the design, commands, results, and
environment; `inputs/` holds the frozen concepts and `outputs/<run>/` every raw
provider response, parsed judgment, score, and report.  `make robustness-tests`
runs their unit tests.

## Notes on the records

- **Polished-slate distances.**  The search stages embedded every local and
  global candidate but not the polished descendants, so
  `<run>-polish-semantic-scores.csv` was produced afterwards by running
  `src/semantic_scores.py` over the preserved polished texts with the same
  pinned embedding model.  Re-running it on the pre-polish concepts reproduces
  the recorded distances exactly.
- **Omitted stage outputs.**  During the run, every searched concept was also
  scored against the 15 seeds, an in-run benchmark the paper does not use.
  Those three files were left out of this repository to keep it smaller.  The
  stage itself is still in the code, so `make run-full` under the reported run
  id would recreate them.
- **Manifest description strings.**  The manifest's
  `population_design.crossover_second_parent` entry describes an earlier
  design ("most semantically distant eligible plan").  The executed rule is the
  one in `pick_recombination_partner` in `src/global_search.py` and in the
  paper: the best-ranked eligible concept that adds a founding lineage the first
  parent lacks, taken from the current parent pool or, failing that, from the
  original seeds.  The released `run_pipeline.py` writes the corrected
  description.
- **Paths.**  Absolute local paths in the manifests, logs, and candidate
  records were replaced with the placeholder `<repository>`, and the Python
  interpreter path with `python`.
- **Raw response ledgers.**  The SQLite call ledgers (about 1.3 GB) are not
  in the repository; every response that shaped the search is nevertheless in
  the CSV records (candidate texts, match analyses, audit verdicts).  The ledgers
  are available from the author on request.
- **Robustness pilots.**  Only the canonical full runs' output directories are
  included; the pilot and dry-run directories described in the robustness
  READMEs were omitted.

## License

The code and records are released under the MIT License (see `LICENSE`).  The
campaign descriptions in `in/projects.csv` are anonymized summaries prepared
for the companion study cited in the paper.
