#!/usr/bin/env python3
"""Estimate request counts, tokens, and OpenRouter charges before a run.

The estimate covers the stages launched by run_pipeline.py (global search,
local search, seed-benchmark scoring).  Polish, the local-only comparison,
reference-set scoring, and the final tournament are separate commands whose
call counts are documented in the README.
"""

import argparse


MODEL_PRICES = {
    "deepseek/deepseek-v3.2": {"input": 0.2072, "output": 0.3108},
}
PRICE_SNAPSHOT_DATE = "2026-08-06"
MODEL_PRICE_SOURCES = {
    "deepseek/deepseek-v3.2": "https://openrouter.ai/deepseek/deepseek-v3.2/pricing",
}

# Empirical token estimates from an earlier 30-firm calibration run.
BASELINE_30 = {
    "input_tokens_m": 29.28,
    "output_tokens_m": 9.83,
    "global_eval_calls": 10500,
    "global_operator_calls": 200,
    "local_calls": 370,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate OpenRouter cost for search runs.")
    parser.add_argument("--firms", type=int, choices=(15,), default=15)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRICES),
        default="deepseek/deepseek-v3.2",
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--common-quality",
        choices=("none", "seed-benchmark"),
        default="seed-benchmark",
        help="Include the shared seed-benchmark quality scoring stage.",
    )
    parser.add_argument(
        "--common-quality-scope",
        choices=("frontier", "all-points"),
        default="all-points",
        help="For seed-benchmark scoring, score only plotted frontier states or every point.",
    )
    return parser.parse_args()


def estimate_commit_round_calls(firms: int, steps: int) -> int:
    """Upper bound for the commit-by-verification round.

    Commit pool <= 5 elites per step + one winner per evaluation + final top-5,
    before text de-duplication; each entry plays 2*firms matches.  The polish
    round adds 5 mutations, 5 audits, and 5 * 2*firms verification matches.
    """
    pool_upper_bound = 5 * steps + (steps + 1) + 5
    polish = 5 + 5 + 5 * 2 * firms
    return pool_upper_bound * 2 * firms + polish


def estimate_common_quality_calls(firms: int, steps: int, scope: str = "frontier") -> int:
    if scope == "frontier":
        local_concept_upper_bound = steps + 1
        global_concept_upper_bound = steps + 1
    else:
        local_concept_upper_bound = steps * 6
        global_concept_upper_bound = firms * (steps + 1)
    concept_upper_bound = local_concept_upper_bound + global_concept_upper_bound
    return concept_upper_bound * firms * 2


def estimate_calls(
    firms: int,
    steps: int,
    common_quality: bool = False,
    common_quality_scope: str = "frontier",
) -> dict[str, int]:
    local_fidelity_calls = steps * 5
    local_calls = steps * (1 + 5 + 6 * 5 + 1)
    global_eval_calls = firms * (firms - 1) * (steps + 1) + (firms + 1) * firms
    global_operator_calls = 2 * (firms // 3) * steps
    global_fidelity_calls = 2 * (firms // 3) * steps
    global_narrator_calls = 2 * steps
    commit_round_calls = estimate_commit_round_calls(firms, steps)
    common_quality_calls = (
        estimate_common_quality_calls(firms, steps, common_quality_scope)
        if common_quality
        else 0
    )
    return {
        "local_calls": local_calls,
        "global_eval_calls": global_eval_calls,
        "global_operator_calls": global_operator_calls,
        "global_narrator_calls": global_narrator_calls,
        "fidelity_audit_calls": local_fidelity_calls + global_fidelity_calls,
        "commit_round_calls": commit_round_calls,
        "common_quality_calls": common_quality_calls,
        "total_calls": (
            local_calls
            + global_eval_calls
            + global_operator_calls
            + global_narrator_calls
            + local_fidelity_calls
            + global_fidelity_calls
            + commit_round_calls
            + common_quality_calls
        ),
    }


def estimate_tokens_m(
    firms: int,
    steps: int,
    common_quality: bool = False,
    common_quality_scope: str = "frontier",
) -> tuple[float, float]:
    eval_ratio = (
        (firms * (firms - 1) * (steps + 1) + (firms + 1) * firms)
        / BASELINE_30["global_eval_calls"]
    )
    operator_ratio = (firms / 30) * (steps / 10)
    local_ratio = steps / 10

    local_input_m = 0.795 * local_ratio
    local_output_m = 0.240 * local_ratio
    global_eval_input_m = 28.061 * eval_ratio
    global_eval_output_m = 9.306 * eval_ratio
    global_operator_input_m = 0.426 * operator_ratio
    global_operator_output_m = 0.282 * operator_ratio

    # Conservative prompt-size allowance for one source+candidate local/mutation
    # audit and two-source+candidate crossover audits.
    local_fidelity_input_m = steps * 5 * 2400 / 1_000_000
    global_fidelity_input_m = steps * ((firms // 3) * 2400 + (firms // 3) * 3500) / 1_000_000
    fidelity_output_m = steps * (5 + 2 * (firms // 3)) * 180 / 1_000_000

    commit_calls = estimate_commit_round_calls(firms, steps)
    commit_input_m = commit_calls * 2672 / 1_000_000
    commit_output_m = commit_calls * 860 / 1_000_000

    common_quality_input_m = 0.0
    common_quality_output_m = 0.0
    if common_quality:
        common_quality_calls = estimate_common_quality_calls(firms, steps, common_quality_scope)
        common_quality_input_m = common_quality_calls * 2672 / 1_000_000
        common_quality_output_m = common_quality_calls * 860 / 1_000_000

    return (
        local_input_m + global_eval_input_m + global_operator_input_m + local_fidelity_input_m + global_fidelity_input_m + commit_input_m + common_quality_input_m,
        local_output_m + global_eval_output_m + global_operator_output_m + fidelity_output_m + commit_output_m + common_quality_output_m,
    )


def main() -> None:
    args = parse_args()
    include_common_quality = args.common_quality == "seed-benchmark"
    calls = estimate_calls(
        args.firms,
        args.steps,
        common_quality=include_common_quality,
        common_quality_scope=args.common_quality_scope,
    )
    input_tokens_m, output_tokens_m = estimate_tokens_m(
        args.firms,
        args.steps,
        common_quality=include_common_quality,
        common_quality_scope=args.common_quality_scope,
    )
    prices = MODEL_PRICES[args.model]
    cost = input_tokens_m * prices["input"] + output_tokens_m * prices["output"]

    print(f"Model: {args.model}")
    print(f"Firms: {args.firms}")
    print(f"Steps: {args.steps}")
    print("Scope: global search, local search, and seed-benchmark scoring (run_pipeline.py)")
    print(f"Estimated calls: {calls['total_calls']:,}")
    print(f"  Local calls: {calls['local_calls']:,}")
    print(f"  Global evaluator calls: {calls['global_eval_calls']:,}")
    print(f"  Global mutation/crossover calls: {calls['global_operator_calls']:,}")
    print(f"  Global narrator calls: {calls['global_narrator_calls']:,}")
    print(f"  Factual-fidelity audit calls: {calls['fidelity_audit_calls']:,}")
    print(f"  Commit-round verification calls (upper bound): {calls['commit_round_calls']:,}")
    if include_common_quality:
        print(f"  Seed-benchmark quality calls ({args.common_quality_scope}): {calls['common_quality_calls']:,}")
    print(f"Estimated tokens: {input_tokens_m:.2f}M input, {output_tokens_m:.2f}M output")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Price snapshot: {PRICE_SNAPSHOT_DATE}")
    if args.model in MODEL_PRICE_SOURCES:
        print(f"Price source: {MODEL_PRICE_SOURCES[args.model]}")


if __name__ == "__main__":
    main()
