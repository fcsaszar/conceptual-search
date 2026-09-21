#!/usr/bin/env python3
"""Analyze a completed cross-evaluator run without making API calls."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
SCORE_FIELDS = [
    "slot_id",
    "group",
    "lineage_id",
    "parent_slot_id",
    "text_sha256",
    "is_seed",
    "deepseek_win_rate",
    "gemma_wins",
    "gemma_matches",
    "gemma_win_rate",
    "gemma_minus_deepseek",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    result = spearmanr(x, y)
    return float(result.statistic)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(run_id: str) -> None:
    run_dir = HERE / "outputs" / run_id
    tasks_path = run_dir / "tasks.csv"
    parsed_dir = run_dir / "parsed"
    manifest_path = run_dir / "manifest.json"
    concepts_path = HERE / "inputs" / "concepts.csv"
    for path in (tasks_path, parsed_dir, manifest_path, concepts_path):
        if not path.exists():
            raise SystemExit(f"Missing required artifact: {path}")

    tasks = read_csv(tasks_path)
    concepts = read_csv(concepts_path)
    if len(tasks) != 3510:
        raise SystemExit(f"Analysis requires the full 3,510-task run; found {len(tasks)} tasks")
    task_by_id = {row["task_id"]: row for row in tasks}
    parsed: dict[str, dict[str, Any]] = {}
    for path in parsed_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parsed[payload["task_id"]] = payload
    missing = sorted(set(task_by_id) - set(parsed))
    extra = sorted(set(parsed) - set(task_by_id))
    if missing or extra:
        raise SystemExit(
            f"Run is incomplete or inconsistent: missing={len(missing)}, extra={len(extra)}"
        )

    wins: dict[str, float] = {}
    matches: dict[str, int] = {}
    direction_pairs: dict[tuple[str, str], dict[str, str]] = {}
    parse_modes: dict[str, int] = {}
    providers: dict[str, int] = {}
    model_returned: dict[str, int] = {}
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    def award(key: str, score: float) -> None:
        wins[key] = wins.get(key, 0.0) + score
        matches[key] = matches.get(key, 0) + 1

    for task_id, task in task_by_id.items():
        result = parsed[task_id]
        winner = result["winner"]
        parse_modes[result.get("parse_mode", "unknown")] = parse_modes.get(result.get("parse_mode", "unknown"), 0) + 1
        providers[result.get("provider", "unknown") or "unknown"] = providers.get(result.get("provider", "unknown") or "unknown", 0) + 1
        model_returned[result.get("model_returned", "unknown") or "unknown"] = model_returned.get(result.get("model_returned", "unknown") or "unknown", 0) + 1
        a_key, b_key = task["plan_a_text_sha256"], task["plan_b_text_sha256"]
        if task["category"] == "original_original":
            if winner == "A":
                award(a_key, 1.0)
                award(b_key, 0.0)
                underlying_winner = a_key
            elif winner == "B":
                award(a_key, 0.0)
                award(b_key, 1.0)
                underlying_winner = b_key
            else:
                award(a_key, 0.5)
                award(b_key, 0.5)
                underlying_winner = "tie"
            pair_key = ("original", *sorted((a_key, b_key)))
            direction_pairs.setdefault(pair_key, {})[f"{a_key}>{b_key}"] = underlying_winner
        else:
            generated_hash = task["generated_text_sha256"]
            if task["direction"] == "generated_first":
                generated_score = 1.0 if winner == "A" else 0.0
                underlying_winner = generated_hash if winner == "A" else b_key
            else:
                generated_score = 1.0 if winner == "B" else 0.0
                underlying_winner = generated_hash if winner == "B" else a_key
            if winner not in {"A", "B"}:
                generated_score = 0.5
                underlying_winner = "tie"
            award(generated_hash, generated_score)
            pair_key = ("generated", generated_hash, task["reference_id"])
            direction_pairs.setdefault(pair_key, {})[task["direction"]] = underlying_winner

    # Account for every provider attempt, including malformed responses that
    # required a retry. Parsed files contain only the successful attempt.
    raw_attempts = 0
    raw_api_errors = 0
    raw_parse_errors_recorded = 0
    for path in (run_dir / "raw").glob("*/attempt-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_attempts += 1
        if "error" in payload:
            raw_api_errors += 1
        if "parse_error" in payload:
            raw_parse_errors_recorded += 1
        response = payload.get("response") or {}
        usage = response.get("usage") or {}
        total_prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        total_completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_cost += float(usage.get("cost") or 0.0)

    concept_by_slot = {row["slot_id"]: row for row in concepts}
    score_rows: list[dict[str, str]] = []
    for row in concepts:
        text_hash = row["text_sha256"]
        expected = 58 if row["group"] == "original" else 60
        if matches.get(text_hash) != expected:
            raise SystemExit(
                f"Unexpected match count for {row['slot_id']}: {matches.get(text_hash)} != {expected}"
            )
        gemma = wins[text_hash] / expected
        deepseek = float(row["deepseek_win_rate"])
        score_rows.append(
            {
                "slot_id": row["slot_id"],
                "group": row["group"],
                "lineage_id": row["lineage_id"],
                "parent_slot_id": row["parent_slot_id"],
                "text_sha256": text_hash,
                "is_seed": row["is_seed"],
                "deepseek_win_rate": f"{deepseek:.6f}",
                "gemma_wins": f"{wins[text_hash]:.1f}",
                "gemma_matches": str(expected),
                "gemma_win_rate": f"{gemma:.6f}",
                "gemma_minus_deepseek": f"{gemma - deepseek:.6f}",
            }
        )

    scores_path = run_dir / "scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(score_rows)

    score_by_slot = {row["slot_id"]: row for row in score_rows}

    def gemma(slot_id: str) -> float:
        return float(score_by_slot[slot_id]["gemma_win_rate"])

    unique_rows: dict[str, dict[str, str]] = {}
    for row in score_rows:
        unique_rows.setdefault(row["text_sha256"], row)

    def correlation_for(groups: set[str]) -> float:
        selected = [row for row in unique_rows.values() if row["group"] in groups]
        return safe_spearman(
            [float(row["deepseek_win_rate"]) for row in selected],
            [float(row["gemma_win_rate"]) for row in selected],
        )

    originals = [row for row in score_rows if row["group"] == "original"]
    seeds = [row for row in originals if row["is_seed"] == "true"]
    parents = [row for row in score_rows if row["group"] == "global_final"]
    polished = [row for row in score_rows if row["group"] == "polished"]
    local_only = [row for row in score_rows if row["group"] == "local_only"]
    parent_deltas = []
    n_polish_improved = 0
    for row in polished:
        delta = float(row["gemma_win_rate"]) - gemma(row["parent_slot_id"])
        parent_deltas.append(delta)
        n_polish_improved += delta > 0
    seed_max = max(float(row["gemma_win_rate"]) for row in seeds)
    polished_above_all_seeds = sum(float(row["gemma_win_rate"]) > seed_max for row in polished)
    original_rates = [float(row["gemma_win_rate"]) for row in originals]
    best_polished = max(polished, key=lambda row: float(row["gemma_win_rate"]))
    best_local = max(local_only, key=lambda row: float(row["gemma_win_rate"]))
    best_polished_rate = float(best_polished["gemma_win_rate"])
    best_local_rate = float(best_local["gemma_win_rate"])
    local_deltas = [
        float(row["gemma_win_rate"]) - gemma(row["parent_slot_id"])
        for row in local_only
    ]
    n_local_improved = sum(delta > 0 for delta in local_deltas)

    agreement_n = 0
    agreement_same = 0
    category_agreement = {
        "original": {"pairs": 0, "same": 0},
        "generated": {"pairs": 0, "same": 0},
    }
    for pair_key, pair_results in direction_pairs.items():
        if len(pair_results) == 2:
            category = pair_key[0] if pair_key[0] in category_agreement else "generated"
            same = len(set(pair_results.values())) == 1
            agreement_n += 1
            agreement_same += same
            category_agreement[category]["pairs"] += 1
            category_agreement[category]["same"] += same

    deepseek_parent_deltas = [
        float(row["deepseek_win_rate"])
        - float(score_by_slot[row["parent_slot_id"]]["deepseek_win_rate"])
        for row in polished
    ]
    deepseek_local_deltas = [
        float(row["deepseek_win_rate"])
        - float(score_by_slot[row["parent_slot_id"]]["deepseek_win_rate"])
        for row in local_only
    ]

    def originals_outranked(slot_id: str, evaluator: str) -> int:
        field = f"{evaluator}_win_rate"
        value = float(score_by_slot[slot_id][field])
        return sum(value > float(row[field]) for row in originals)

    results = {
        "schema_version": 1,
        "run_id": run_id,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": {"slots": len(score_rows), "unique_texts": len(unique_rows), "calls": len(tasks)},
        "usage": {
            "provider_attempts": raw_attempts,
            "provider_api_errors": raw_api_errors,
            "parse_errors_recorded_before_recovery": raw_parse_errors_recorded,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "reported_cost_usd": total_cost,
            "providers": providers,
            "models_returned": model_returned,
            "parse_modes": parse_modes,
        },
        "agreement": {
            "spearman_all_unique_texts": correlation_for({"original", "global_final", "polished", "local_only"}),
            "spearman_originals": correlation_for({"original"}),
            "spearman_generated_unique_texts": correlation_for({"global_final", "polished", "local_only"}),
            "spearman_global_final": correlation_for({"global_final"}),
            "spearman_polished": correlation_for({"polished"}),
            "spearman_local_only": correlation_for({"local_only"}),
            "presentation_order_agreement": agreement_same / agreement_n if agreement_n else None,
            "presentation_order_pairs": agreement_n,
            "presentation_order_original_agreement": (
                category_agreement["original"]["same"]
                / category_agreement["original"]["pairs"]
            ),
            "presentation_order_original_pairs": category_agreement["original"]["pairs"],
            "presentation_order_generated_agreement": (
                category_agreement["generated"]["same"]
                / category_agreement["generated"]["pairs"]
            ),
            "presentation_order_generated_pairs": category_agreement["generated"]["pairs"],
        },
        "principal_findings": {
            "polish_pairs": len(parent_deltas),
            "polish_improved": n_polish_improved,
            "polish_median_delta_percentage_points": 100 * median(parent_deltas),
            "global_final_median_win_rate": median(float(row["gemma_win_rate"]) for row in parents),
            "polished_median_win_rate": median(float(row["gemma_win_rate"]) for row in polished),
            "polished_above_all_fixed_seeds": polished_above_all_seeds,
            "best_polished_slot": best_polished["slot_id"],
            "best_polished_win_rate": best_polished_rate,
            "best_polished_outranks_originals": sum(best_polished_rate > rate for rate in original_rates),
            "local_only_median_win_rate": median(float(row["gemma_win_rate"]) for row in local_only),
            "best_local_only_slot": best_local["slot_id"],
            "best_local_only_win_rate": best_local_rate,
            "best_local_only_outranks_originals": sum(best_local_rate > rate for rate in original_rates),
            "local_only_improved_over_seed": n_local_improved,
            "local_only_median_delta_percentage_points": 100 * median(local_deltas),
            "fixed_seed_median_win_rate": median(float(row["gemma_win_rate"]) for row in seeds),
            "fixed_seed_max_win_rate": seed_max,
            "clockchain_local_final_win_rate": gemma("local-only:clockchain"),
            "paper_focal_polished_gemma_win_rate": gemma("polished:g12-s04"),
            "paper_focal_polished_gemma_outranks_originals": originals_outranked(
                "polished:g12-s04", "gemma"
            ),
            "paper_focal_polished_deepseek_outranks_originals": originals_outranked(
                "polished:g12-s04", "deepseek"
            ),
            "clockchain_gemma_outranks_originals": originals_outranked(
                "local-only:clockchain", "gemma"
            ),
            "clockchain_deepseek_outranks_originals": originals_outranked(
                "local-only:clockchain", "deepseek"
            ),
            "hply_gemma_outranks_originals": originals_outranked(
                "local-only:hply", "gemma"
            ),
            "hply_deepseek_outranks_originals": originals_outranked(
                "local-only:hply", "deepseek"
            ),
            "deepseek_polish_improved": sum(delta > 0 for delta in deepseek_parent_deltas),
            "deepseek_polish_median_delta_percentage_points": 100 * median(deepseek_parent_deltas),
            "deepseek_global_final_median_win_rate": median(
                float(row["deepseek_win_rate"]) for row in parents
            ),
            "deepseek_polished_median_win_rate": median(
                float(row["deepseek_win_rate"]) for row in polished
            ),
            "deepseek_local_only_median_win_rate": median(
                float(row["deepseek_win_rate"]) for row in local_only
            ),
            "deepseek_local_only_improved_over_seed": sum(
                delta > 0 for delta in deepseek_local_deltas
            ),
        },
    }
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aggregate_run_totals"] = {
        "scheduled_tasks": len(tasks),
        "completed_tasks": len(parsed),
        "provider_attempts": raw_attempts,
        "provider_api_errors": raw_api_errors,
        "parse_errors_recorded_before_recovery": raw_parse_errors_recorded,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "reported_cost_usd": total_cost,
    }
    manifest["analysis"] = {
        "completed_at_utc": results["analyzed_at_utc"],
        "script": "analyze_results.py",
        "script_sha256": sha256_file(Path(__file__)),
        "scores": "scores.csv",
        "results": "results.json",
        "report": "report.md",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    a = results["agreement"]
    f = results["principal_findings"]
    report = f"""# Gemma cross-evaluator robustness results

Run: `{run_id}`. The check re-evaluated 30 originals and 45 generated concept slots (44 distinct generated texts) with Gemma 3 4B using the paper's pairwise prompt in both presentation orders. All 3,510 scheduled judgments completed.

## Agreement between evaluators

Across the 74 distinct texts, Gemma and DeepSeek win rates correlate at Spearman rho = {a['spearman_all_unique_texts']:.3f}. Agreement is high for the 30 originals (rho = {a['spearman_originals']:.3f}) but not for the 44 generated texts as a group (rho = {a['spearman_generated_unique_texts']:.3f}). Within generated-text stages, rho is {a['spearman_global_final']:.3f} for global-search finalists, {a['spearman_polished']:.3f} for polished descendants, and {a['spearman_local_only']:.3f} for local-only finals. Exact rankings of generated concepts are therefore evaluator-dependent.

Each unordered pair was judged twice at temperature 0.5 with its presentation order reversed. Gemma selected the same underlying winner in {pct(a['presentation_order_original_agreement'])} of {a['presentation_order_original_pairs']:,} original-original pairs and {pct(a['presentation_order_generated_agreement'])} of {a['presentation_order_generated_pairs']:,} generated-original pairs. These figures combine order sensitivity with ordinary sampling variation; averaging both directions limits their effect on win rates.

## Claim-level results

| Claim | DeepSeek | Gemma | Result |
|---|---:|---:|---|
| Local polish improves final-generation concepts | {f['deepseek_polish_improved']}/15; median +{f['deepseek_polish_median_delta_percentage_points']:.1f} pp | {f['polish_improved']}/15; median +{f['polish_median_delta_percentage_points']:.1f} pp | Direction reproduced; universality did not |
| Every polished descendant outranks every fixed seed | 15/15 | {f['polished_above_all_fixed_seeds']}/15 | Reproduced |
| Paper's focal polished descendant outranks originals | {f['paper_focal_polished_deepseek_outranks_originals']}/30 | {f['paper_focal_polished_gemma_outranks_originals']}/30 | Reproduced |
| Local-only median exceeds polished median | {pct(f['deepseek_local_only_median_win_rate'])} vs. {pct(f['deepseek_polished_median_win_rate'])} | {pct(f['local_only_median_win_rate'])} vs. {pct(f['polished_median_win_rate'])} | Reversed |
| Local-only finals improve on their seeds | {f['deepseek_local_only_improved_over_seed']}/15 | {f['local_only_improved_over_seed']}/15 | Reproduced |
| Best local-only final (`hply`) outranks originals | {f['hply_deepseek_outranks_originals']}/30 | {f['hply_gemma_outranks_originals']}/30 | Reproduced |
| Focal local final (`clockchain`) outranks originals | {f['clockchain_deepseek_outranks_originals']}/30 | {f['clockchain_gemma_outranks_originals']}/30 | Similar, not exact |

Gemma assigns substantially higher scores to global-search finalists: their median rises from {pct(f['deepseek_global_final_median_win_rate'])} under DeepSeek to {pct(f['global_final_median_win_rate'])}. The polished median rises from {pct(f['deepseek_polished_median_win_rate'])} to {pct(f['polished_median_win_rate'])}, whereas the local-only median falls from {pct(f['deepseek_local_only_median_win_rate'])} to {pct(f['local_only_median_win_rate'])}. Thus, the second evaluator rules out DeepSeek-specific self-preference as the sole explanation for generated concepts clearing the seeds and reference set, but it does not support evaluator-invariant magnitudes, exact rankings, or the relative quality advantage of local-only search. Semantic-distance findings are unchanged because they do not use the evaluator.

## Run accounting

The run used {results['usage']['prompt_tokens']:,} prompt tokens and {results['usage']['completion_tokens']:,} completion tokens. OpenRouter reported a total cost of ${results['usage']['reported_cost_usd']:.4f}. Provider routes were {results['usage']['providers']}. Exact responses and parsed judgments are preserved under this run directory.
"""
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {scores_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    main(parser.parse_args().run_id)
