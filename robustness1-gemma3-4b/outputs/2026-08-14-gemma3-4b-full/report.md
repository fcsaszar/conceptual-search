# Gemma cross-evaluator robustness results

Run: `2026-08-14-gemma3-4b-full`. The check re-evaluated 30 originals and 45 generated concept slots (44 distinct generated texts) with Gemma 3 4B using the paper's pairwise prompt in both presentation orders. All 3,510 scheduled judgments completed.

## Agreement between evaluators

Across the 74 distinct texts, Gemma and DeepSeek win rates correlate at Spearman rho = 0.384. Agreement is high for the 30 originals (rho = 0.938) but not for the 44 generated texts as a group (rho = 0.017). Within generated-text stages, rho is 0.425 for global-search finalists, 0.130 for polished descendants, and 0.812 for local-only finals. Exact rankings of generated concepts are therefore evaluator-dependent.

Each unordered pair was judged twice at temperature 0.5 with its presentation order reversed. Gemma selected the same underlying winner in 38.6% of 435 original-original pairs and 55.6% of 1,320 generated-original pairs. These figures combine order sensitivity with ordinary sampling variation; averaging both directions limits their effect on win rates.

## Claim-level results

| Claim | DeepSeek | Gemma | Result |
|---|---:|---:|---|
| Local polish improves final-generation concepts | 15/15; median +28.3 pp | 13/15; median +5.0 pp | Direction reproduced; universality did not |
| Every polished descendant outranks every fixed seed | 15/15 | 15/15 | Reproduced |
| Paper's focal polished descendant outranks originals | 24/30 | 29/30 | Reproduced |
| Local-only median exceeds polished median | 68.3% vs. 63.3% | 55.0% vs. 83.3% | Reversed |
| Local-only finals improve on their seeds | 15/15 | 15/15 | Reproduced |
| Best local-only final (`hply`) outranks originals | 28/30 | 28/30 | Reproduced |
| Focal local final (`clockchain`) outranks originals | 28/30 | 26/30 | Similar, not exact |

Gemma assigns substantially higher scores to global-search finalists: their median rises from 36.7% under DeepSeek to 80.0%. The polished median rises from 63.3% to 83.3%, whereas the local-only median falls from 68.3% to 55.0%. Thus, the second evaluator rules out DeepSeek-specific self-preference as the sole explanation for generated concepts clearing the seeds and reference set, but it does not support evaluator-invariant magnitudes, exact rankings, or the relative quality advantage of local-only search. Semantic-distance findings are unchanged because they do not use the evaluator.

## Run accounting

The run used 7,618,931 prompt tokens and 2,137,894 completion tokens. OpenRouter reported a total cost of $0.5947. Provider routes were {'DeepInfra': 3510}. Exact responses and parsed judgments are preserved under this run directory.
