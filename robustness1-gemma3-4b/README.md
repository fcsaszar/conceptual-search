# Cross-evaluator robustness check

This folder tests whether the paper's main ordinal results depend on using DeepSeek V3.2 as both generator and evaluator. It does not rerun either search. It freezes the concepts already produced by the canonical run and measures them again with Gemma 3 4B Instruct, an independently developed open-weight evaluator.

## Why Gemma 3 4B

Gemma 3 4B was the strongest non-DeepSeek open-weight model in the companion forecasting tournament: its ranking of these same 30 ventures correlated at 0.55 with realized fundraising and correctly ordered about 71 percent of pairs. It therefore supplies task-specific prior validation rather than being chosen after seeing the robustness results. The model's weights are public, its August 2024 knowledge cutoff precedes the campaigns, and its size makes local replication practical. The run uses OpenRouter model `google/gemma-3-4b-it`; the corresponding Hugging Face repository is `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`.

Open weights make the analysis replicable in principle, but API execution is not bit-for-bit deterministic across providers or hardware. The run manifest therefore records the requested model, returned model, upstream provider when supplied by OpenRouter, prompt, decoding settings, token use, cost, software environment, and every raw response. A future local replication should use the recorded weights revision and document its inference engine, numerical precision, and quantization.

## Sample and estimand

The check contains 75 concept slots: 30 originals, 15 concepts from global search's final generation, their 15 polished descendants, and 15 local-only finals. Two final-generation slots contain the same text, leaving 44 distinct generated texts and 74 distinct texts overall. The duplicate is measured once and assigned the same score in both slots.

Each generated text faces all 30 originals twice, once in each presentation order. The originals also complete their own double round-robin under Gemma. The design therefore requires (30 \times 29 = 870) original-versus-original calls and (44 \times 30 \times 2 = 2,640) generated-versus-original calls, for 3,510 calls in total. Generated concepts are not compared with one another because the paper defines performance as win rate against the 30-venture reference set.

The primary test is whether the ordinal findings survive under the second evaluator: local polish improves global-search finalists, polished concepts clear the fixed seed set, and local-only search retains its quality advantage while global recombination supplies semantic distance. The semantic-distance results do not need to be recomputed because they do not use the evaluator.

## Files

- `prompt.py` freezes the exact pairwise prompt and criteria used in the paper.
- `model-metadata.json` records the OpenRouter catalog entry, public weights revision, knowledge cutoff, and prices observed before the run.
- `run-code-hashes.json` records the exact runner versions used before and after the documented parser correction.
- `prepare_inputs.py` extracts and validates the concepts from the canonical run, attaches their canonical DeepSeek scores, and writes `inputs/concepts.csv` plus a source-hash manifest.
- `run_robustness.py` builds the 3,510 ordered comparisons and runs them through OpenRouter. It is resumable and preserves each raw response before parsing it.
- `analyze_results.py` computes Gemma win rates, correlations with DeepSeek, presentation-order agreement, and the main ordinal findings without making API calls.
- `test_robustness.py` checks the frozen prompt, parser, sample, deduplication, and call arithmetic.
- `outputs/<run-id>/raw/` contains one directory per task and one JSON file per provider attempt.
- `outputs/<run-id>/parsed/` contains one normalized judgment per completed task.
- `outputs/<run-id>/manifest.json`, `tasks.csv`, `scores.csv`, `results.json`, `report.md`, and `logs/run.log` document and summarize the run. The full run also preserves `sample-manifest-at-run.json`, whose hash matches the input-manifest hash recorded at run start.

This repository includes the output directory of the canonical full run only; the pilot and dry-run directories are omitted.

## Environment

Python requires the packages in `requirements.txt`. A live run also requires `OPENROUTER_API_KEY` in the environment. The key is read at runtime and is never written to an output file.

```bash
cd robustness1-gemma3-4b
python3 -m pip install -r requirements.txt
```

## Prepare and validate

Run input preparation whenever the canonical source files change. The program stops if the frozen evaluator prompt differs from the one used by the main analysis, if the expected groups or seed ids change, or if the duplicate count changes.

```bash
python3 prepare_inputs.py
python3 -m unittest -v test_robustness.py
python3 run_robustness.py --run-id dry-run --dry-run
```

The dry run writes the complete task list and run manifest but makes no API calls.

## Live pilot and full run

The pipeline first runs a stratified pilot that covers original comparisons and all three generated-concept groups. The pilot exercises the real model, prompt, parser, raw-response storage, and provider accounting.

```bash
python3 run_robustness.py --run-id 2026-08-14-gemma3-4b-pilot --limit 16 --max-concurrency 8
```

After inspecting the pilot's failures, latency, and reported cost, run the full design:

```bash
python3 run_robustness.py --run-id 2026-08-14-gemma3-4b-full --max-concurrency 40
python3 analyze_results.py --run-id 2026-08-14-gemma3-4b-full
```

The live defaults match the paper's evaluator settings: temperature 0.5, 2,000 maximum response tokens, the unchanged pairwise prompt, and both presentation orders. The Gemma run additionally sends random seed 42 because the endpoint exposes that control. No API-level structured-output constraint is imposed because the original evaluation relied on the prompt's JSON instruction.

## Completed run and results

The canonical robustness run is `2026-08-14-gemma3-4b-full`. It completed all 3,510 scheduled judgments through OpenRouter, which routed the successful responses to DeepInfra. Across the full run and its safe resume, the software made 3,565 provider attempts, encountered one API error, used 7,618,931 prompt tokens and 2,137,894 completion tokens, and incurred $0.5947 in provider-reported charges. The separate 16-call pilot cost $0.0028. The full run took 34 minutes of wall-clock time, including the parser inspection described below.

The result is supportive but not uniform. Gemma and DeepSeek rankings correlate at Spearman rho = 0.938 for the 30 originals. Every polished descendant still outranks every fixed seed, and the paper's focal polished descendant outranks 29 of the 30 originals under Gemma, compared with 24 under DeepSeek. Gemma also finds that 13 of 15 polish operations improve their parents and that all 15 local-only finals improve their seeds. The best local-only final outranks the same 28 originals under both evaluators.

The evaluators do not agree on every inference. Their scores correlate at only rho = 0.017 across the 44 generated texts considered together. Most importantly, Gemma reverses the relative quality advantage of local-only search: its median is 55.0 percent, against 83.3 percent for the polished slate, whereas DeepSeek scores the two at 68.3 and 63.3 percent. Gemma also assigns the unpolished global-search finalists a much higher median score than DeepSeek (80.0 versus 36.7 percent). The check therefore shows that DeepSeek-specific self-preference cannot by itself explain why generated concepts clear the seeds and reference set, but exact generated-concept rankings, magnitudes, and the local-only-versus-polished quality comparison remain evaluator-dependent. Semantic-distance findings do not use either evaluator and are unchanged.

The concise claim-by-claim comparison is in `outputs/2026-08-14-gemma3-4b-full/report.md`; machine-readable estimates are in `results.json`, and concept-level scores are in `scores.csv`. The run directory also preserves the task list, manifest, logs, normalized judgments, and every raw provider response. These files are the source of record for any result reported in the paper.

## Resuming and recovering

Re-run the exact same command with the same run id. A task is complete only when `parsed/<task-id>.json` exists. Existing parsed judgments are skipped. Before making another paid call, the software re-parses every saved raw response; this allows a transparent parser correction to recover an explicit winner without discarding or regenerating the original response. If no saved response contains a valid or explicitly recoverable winner, the task can make further attempts up to the recorded limit. Per-task lock files prevent simultaneous processes from issuing the same call; locks older than six hours are treated as stale.

Gemma often encloses its response in a Markdown JSON fence. In some responses it also writes an invalid `analysis` object while still supplying exactly one explicit `"winner": "A"` or `"winner": "B"` field. The parser accepts standard JSON, removes an outer Markdown fence, normalizes `"Project A"` or `"Project B"`, and, if the surrounding JSON is malformed, recovers only a single explicit winner field. It never infers the winner from the prose. The raw response and recovery mode remain recorded for audit.

The default failure budgets stop the run cleanly if reported cost exceeds $5, ten tasks fail consecutively, the final task-failure rate exceeds 1 percent after 100 tasks, or the response-level parse-failure rate exceeds 10 percent after 100 attempts. A stopped run remains resumable. Review malformed responses before relaxing a threshold.

## Interpretation

The check reduces the specific concern that DeepSeek alone favors concepts carrying its own stylistic fingerprint. It does not convert model judgment into market evidence, establish that either evaluator is correct outside this setting, or rule out preferences shared by language models. The disagreement over generated-text rankings is itself evidence that these limits should remain explicit in the paper.
