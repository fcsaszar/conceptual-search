# Qwen cross-evaluator robustness results

Run: `2026-08-14-qwen2.5-72b-full`. The check re-evaluated 30 originals and 45 generated concept slots (44 distinct generated texts) with Qwen 2.5 72B using the paper's pairwise prompt in both presentation orders. All 3,510 scheduled judgments completed.

## Agreement between evaluators

Across the 74 distinct texts, Qwen and DeepSeek win rates correlate at Spearman rho = 0.592. The correlation is 0.818 for the 30 originals and 0.474 for the 44 generated texts as a group. Within generated-text stages, rho is 0.085 for global-search finalists, 0.338 for polished descendants, and 0.779 for local-only finals. On the 30 originals, Qwen's ranking correlates at 0.511 with realized fundraising, compared with 0.664 for DeepSeek under this measure.

Each unordered pair was judged twice at temperature 0.5 with its presentation order reversed. Qwen selected the same underlying winner in 76.3% of 435 original-original pairs and 62.5% of 1,320 generated-original pairs. These figures combine order sensitivity with ordinary sampling variation; averaging both directions limits their effect on win rates.

## Claim-level results

| Claim | DeepSeek | Qwen | Result |
|---|---:|---:|---|
| Local polish improves final-generation concepts | 15/15; median +28.3 pp | 12/15; median +8.3 pp | Direction reproduced; universality did not |
| Every polished descendant outranks every fixed seed | 15/15 | 9/15 | Weaker |
| Paper's focal polished descendant outranks originals | 24/30 | 26/30 | Stronger |
| Local-only median exceeds polished median | 68.3% vs. 63.3% | 63.3% vs. 80.0% | Reversed |
| Local-only finals improve on their seeds | 15/15 | 15/15 | Reproduced |
| DeepSeek's best local-only final (`hply`) outranks originals | 28/30 | 29/30 | Stronger |
| Focal local final (`clockchain`) outranks originals | 28/30 | 23/30 | Weaker |

The global-final median is 70.0% under Qwen and 36.7% under DeepSeek. The corresponding polished medians are 80.0% and 63.3%; the local-only medians are 63.3% and 68.3%. Semantic-distance findings are unchanged because they do not use the evaluator.

## Run accounting

The run used 7,366,709 prompt tokens and 2,817,333 completion tokens. OpenRouter reported a total cost of $3.7774. Provider routes were {'DeepInfra': 3510}. Exact responses and parsed judgments are preserved under this run directory.
