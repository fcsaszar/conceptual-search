# Qwen 2.5 72B cross-evaluator robustness check

This folder tests the paper's evaluator-relative findings with Qwen 2.5 72B Instruct. It does not rerun any search. It freezes the concepts produced by the canonical DeepSeek run and measures them again with a substantially larger open-weight model from another developer. The preceding Gemma 3 4B check remains intact in `../robustness1-gemma3-4b/`.

## Why Qwen 2.5 72B

The Gemma check strongly reproduced the ranking of the 30 original ventures but changed the ordering of generated concepts and reversed the relative median quality of the local-only and polished slates. One possible explanation is that a 4B-parameter evaluator has difficulty distinguishing the longer and less conventional generated concepts. Qwen 2.5 72B provides a more demanding test: it has 72.7 billion parameters, comes from a different model family, has open weights, and has a June 2024 knowledge cutoff that precedes every campaign. Its model card emphasizes instruction following and structured-output performance, both relevant to the fixed pairwise prompt.

Qwen was not included in the companion prospective tournament. The design therefore evaluates its task-specific validity rather than assuming it: Qwen re-ranks all 30 originals, allowing its ranking to be compared with DeepSeek and realized fundraising before interpreting its judgments of generated concepts. The model was selected before seeing any Qwen results, based on capacity, model-family independence, open weights, and a pre-campaign cutoff.

The live run uses OpenRouter model `qwen/qwen-2.5-72b-instruct`; the public weights are in `Qwen/Qwen2.5-72B-Instruct`, revision `495f39366efef23836d0cfae4fbe635880d2be31`. The OpenRouter route is pinned to DeepInfra, the route that successfully served the model in the pilot, rather than left to automatic routing. OpenRouter execution is not bit-for-bit deterministic across hardware. The run manifest records the requested and returned models, provider route, prompt and input hashes, decoding settings, token use, cost, software environment, and every raw response. A future local replication should record its inference engine, numerical precision, and quantization.

## Sample and estimand

The check contains the same 75 concept slots used in the Gemma check: 30 originals, 15 concepts from global search's final generation, their 15 polished descendants, and 15 local-only finals. Two global-final slots contain the same text, leaving 44 distinct generated texts and 74 distinct texts overall. The duplicate is measured once and assigned the same score in both slots.

Each generated text faces all 30 originals twice, once in each presentation order. The originals also complete their own double round-robin under Qwen. The design therefore requires 870 original-versus-original calls and 2,640 generated-versus-original calls, for 3,510 calls in total. Generated concepts are not compared with one another because the paper defines performance as win rate against the 30-venture reference set.

The analysis asks which DeepSeek findings survive under Qwen, whether Qwen agrees more closely with DeepSeek than Gemma did on unusual generated concepts, and whether the evaluators agree on the relative median quality of local-only and globally recombined-then-polished search. Semantic distances are not recomputed because they do not use the evaluator.

## Files

- `prompt.py` freezes the exact pairwise prompt and criteria used in the paper and the Gemma check.
- `model-metadata.json` records the OpenRouter catalog entry, public weights revision, cutoff, parameter count, and prices observed before the run.
- `prepare_inputs.py` extracts and validates the concepts from the canonical search, attaches canonical DeepSeek scores and realized fundraising for the originals, and writes a source-hash manifest.
- `run_robustness.py` builds the 3,510 ordered comparisons and calls Qwen through OpenRouter. It is resumable and preserves every provider response before parsing it.
- `analyze_results.py` computes Qwen win rates, correlations with DeepSeek and realized fundraising, presentation-order agreement, and the claim-level comparisons without making API calls.
- `compare_evaluators.py` joins the Qwen scores to the preserved Gemma run and reports a three-evaluator comparison.
- `test_robustness.py` checks the frozen prompt, parser, sample, deduplication, and call arithmetic.
- `outputs/<run-id>/raw/` stores every provider attempt; `parsed/` stores one normalized judgment per completed task.
- `outputs/<run-id>/manifest.json`, `tasks.csv`, `scores.csv`, `results.json`, `report.md`, and `logs/run.log` document and summarize each run.

This repository includes the output directory of the canonical full run only; the pilot and dry-run directories are omitted.

## Environment and commands

Python requires the packages in `requirements.txt`. A live run also requires `OPENROUTER_API_KEY`; the key is read at runtime and is never written to disk.

```bash
cd robustness2-qwen2.5-72b
python3 -m pip install -r requirements.txt
python3 prepare_inputs.py
python3 -m unittest -v test_robustness.py
python3 run_robustness.py --run-id dry-run --dry-run
```

Run the real model first on a stratified pilot that covers original comparisons and every generated-concept group. Inspect latency, parsing, failures, and projected cost before scaling.

```bash
python3 run_robustness.py --run-id 2026-08-14-qwen2.5-72b-pilot2 --limit 16 --max-concurrency 8
python3 run_robustness.py --run-id 2026-08-14-qwen2.5-72b-full --max-concurrency 40 --max-attempts 20 --max-consecutive-failures 40 --max-cost 8
python3 analyze_results.py --run-id 2026-08-14-qwen2.5-72b-full
python3 compare_evaluators.py --qwen-run-id 2026-08-14-qwen2.5-72b-full
```

The live defaults match the other evaluators: temperature 0.5, at most 2,000 response tokens, random seed 42 where supported, the unchanged pairwise prompt, and both presentation orders. No API-level structured-output constraint is imposed because the paper's evaluation relied on the prompt's JSON instruction.

## Preflight record

The preflight artifacts are retained because they document why the provider route was frozen. The first pilot was blocked by the execution sandbox and reached no provider. Automatic OpenRouter routing in the second pilot completed all judgments but intermittently returned error objects from an incompatible route. DeepInfra was then temporarily overloaded, and NovitaAI reported no available endpoint. Once DeepInfra recovered, the one-call check succeeded; the final 16-task check completed despite 13 transient overload responses. Provider errors incurred no charge.

| Run | Completed | Attempts | Provider errors | Recorded cost |
|---|---:|---:|---:|---:|
| `2026-08-14-qwen2.5-72b-pilot` | 0/16 | 76 | 76 | $0.0000 |
| `2026-08-14-qwen2.5-72b-pilot2` | 16/16 | 31 | 15 | $0.0162 |
| `2026-08-14-qwen2.5-72b-pilot3` | 0/16 | 80 | 80 | $0.0000 |
| `2026-08-14-qwen2.5-72b-pilot4-novita` | 0/4 | 20 | 20 | $0.0000 |
| `2026-08-14-qwen2.5-72b-pilot5-deepinfra` | 1/1 | 1 | 0 | $0.0010 |
| `2026-08-14-qwen2.5-72b-pilot6-deepinfra` | 16/16 | 29 | 13 | $0.0166 |

## Completed canonical run

The canonical run `2026-08-14-qwen2.5-72b-full` completed all 3,510 judgments on August 14, 2026. It required 3,512 provider calls because two responses contained malformed JSON and were retried. No task failed and no API call failed. The run used 7,366,709 prompt tokens and 2,817,333 completion tokens, took 47.4 minutes, and cost $3.7774. DeepInfra served every accepted judgment.

Qwen and DeepSeek rankings correlate at Spearman $\rho=.818$ for the 30 originals and $\rho=.474$ for the 44 distinct generated concepts. Qwen's ranking of the originals correlates at $\rho=.511$ with realized fundraising, compared with $.664$ for DeepSeek and $.597$ for Gemma in this frozen sample. This supports Qwen's use as an informative outside-family judge, though not as an evaluator equivalent to DeepSeek.

The paper's focal polished concept is robust: it outranks 26 of the 30 originals under Qwen, compared with 24 under DeepSeek and 29 under Gemma. Qwen also finds that local refinement improves 12 of the 15 global-search finalists, with a median gain of 8.3 percentage points; Gemma finds 13 of 15 and a 5.0-point median gain. Qwen reproduces improvement for all 15 local-only searches, and its best local-only final outranks 29 originals.

Two exact DeepSeek conclusions do not generalize across judges. Only nine polished descendants outrank every fixed seed under Qwen, although all fifteen do under DeepSeek and Gemma. Qwen and Gemma also place the polished slate's median above the local-only slate's median, reversing DeepSeek's local-only advantage. Thus, the high placement of the focal output and the broad value of refinement survive cross-family evaluation; universal and slate-level comparisons remain evaluator-relative. See the canonical run's `report.md` and `cross-evaluator-comparison.md` for the full results.

## Resuming and recovery

Re-run the same command with the same run id. A task is complete only when `parsed/<task-id>.json` exists; completed tasks are skipped. Before making another paid call, the software re-parses saved raw responses. A malformed response is never discarded merely because the parser improves later.

The parser accepts exact JSON, removes an outer Markdown JSON fence, and normalizes `Project A` or `Project B`. If the surrounding JSON is malformed, it recovers a winner only when the response contains exactly one explicit quoted `winner` field. It never infers the winner from prose. Per-task locks prevent duplicate concurrent calls and expire after six hours.

The run stops cleanly if cost exceeds the specified cap, ten tasks fail consecutively, the final task-failure rate exceeds one percent after 100 tasks, or the response-level parse-failure rate exceeds ten percent after 100 attempts. A stopped run remains resumable. Review raw failures before changing any threshold.

## Interpretation

Agreement with DeepSeek and Gemma on the generated concepts would strengthen the claim that the paper's principal ordinal findings do not depend on one model's stylistic preferences. Disagreement would show which conclusions are evaluator-dependent. None of these outcomes converts model judgment into market evidence or rules out preferences shared across language-model families.
