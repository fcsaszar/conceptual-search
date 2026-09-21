# Three-evaluator comparison

This report compares the paper's DeepSeek scores with two cross-family, open-weight evaluators. All three columns refer to the same frozen concept texts; Gemma and Qwen used the same pairwise prompt and both presentation orders.

| Subset | N | DeepSeek--Gemma | DeepSeek--Qwen | Gemma--Qwen |
|---|---:|---:|---:|---:|
| All Unique Texts | 74 | 0.384 | 0.592 | 0.733 |
| Originals | 30 | 0.938 | 0.818 | 0.869 |
| All Generated Unique Texts | 44 | 0.017 | 0.474 | 0.531 |
| Global Final | 14 | 0.425 | 0.085 | 0.222 |
| Polished | 15 | 0.130 | 0.338 | 0.467 |
| Local Only | 15 | 0.812 | 0.779 | 0.741 |

Spearman correlations with realized fundraising among the 30 originals are 0.664 for DeepSeek, 0.597 for Gemma, and 0.511 for Qwen. These correlations are descriptive checks on the reference set, not estimates of evaluator accuracy for generated concepts.

| Quantity | DeepSeek | Gemma | Qwen |
|---|---:|---:|---:|
| Polish Improved Count | 15.000 | 13.000 | 12.000 |
| Polish Median Delta Percentage Points | 28.340 | 5.000 | 8.333 |
| Polished Above All Fixed Seeds | 15.000 | 15.000 | 9.000 |
| Focal Polished Outranks Originals | 24.000 | 29.000 | 26.000 |
| Local Only Median Win Rate | 0.683 | 0.550 | 0.633 |
| Polished Median Win Rate | 0.633 | 0.833 | 0.800 |
| Hply Outranks Originals | 28.000 | 28.000 | 29.000 |

Win-rate levels need not be comparable across evaluators because each judge can apply a different threshold. The most informative checks are whether rankings and the paper's ordinal conclusions survive.
