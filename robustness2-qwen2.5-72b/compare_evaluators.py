#!/usr/bin/env python3
"""Compare DeepSeek, Gemma, and Qwen scores without making API calls."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
DEFAULT_GEMMA_RUN = (
    HERE.parent
    / "robustness1-gemma3-4b"
    / "outputs"
    / "2026-08-14-gemma3-4b-full"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rho(rows: list[dict[str, str]], first: str, second: str) -> float:
    result = spearmanr(
        [float(row[first]) for row in rows],
        [float(row[second]) for row in rows],
    )
    return float(result.statistic)


def main(qwen_run_id: str, gemma_run: Path) -> None:
    qwen_run = HERE / "outputs" / qwen_run_id
    qwen_scores_path = qwen_run / "scores.csv"
    gemma_scores_path = gemma_run / "scores.csv"
    qwen_results_path = qwen_run / "results.json"
    gemma_results_path = gemma_run / "results.json"
    for path in (
        qwen_scores_path,
        gemma_scores_path,
        qwen_results_path,
        gemma_results_path,
    ):
        if not path.exists():
            raise SystemExit(f"Missing required artifact: {path}")

    qwen_slots = read_csv(qwen_scores_path)
    gemma_slots = read_csv(gemma_scores_path)
    gemma_by_slot = {row["slot_id"]: row for row in gemma_slots}
    if set(gemma_by_slot) != {row["slot_id"] for row in qwen_slots}:
        raise SystemExit("Qwen and Gemma runs do not contain the same concept slots")

    joined_slots: list[dict[str, str]] = []
    for row in qwen_slots:
        gemma = gemma_by_slot[row["slot_id"]]
        if gemma["text_sha256"] != row["text_sha256"]:
            raise SystemExit(f"Text hash mismatch for {row['slot_id']}")
        joined_slots.append({**row, "gemma_win_rate": gemma["gemma_win_rate"]})

    unique_by_hash: dict[str, dict[str, str]] = {}
    for row in joined_slots:
        unique_by_hash.setdefault(row["text_sha256"], row)
    unique_rows = list(unique_by_hash.values())

    subsets = {
        "all_unique_texts": {"original", "global_final", "polished", "local_only"},
        "originals": {"original"},
        "all_generated_unique_texts": {"global_final", "polished", "local_only"},
        "global_final": {"global_final"},
        "polished": {"polished"},
        "local_only": {"local_only"},
    }
    correlations = {}
    for name, groups in subsets.items():
        selected = [row for row in unique_rows if row["group"] in groups]
        correlations[name] = {
            "n": len(selected),
            "deepseek_gemma": rho(selected, "deepseek_win_rate", "gemma_win_rate"),
            "deepseek_qwen": rho(selected, "deepseek_win_rate", "qwen_win_rate"),
            "gemma_qwen": rho(selected, "gemma_win_rate", "qwen_win_rate"),
        }

    originals = [row for row in joined_slots if row["group"] == "original"]
    fundraising_correlations = {
        "n": len(originals),
        "deepseek": rho(originals, "deepseek_win_rate", "realized_raised_usd"),
        "gemma": rho(originals, "gemma_win_rate", "realized_raised_usd"),
        "qwen": rho(originals, "qwen_win_rate", "realized_raised_usd"),
    }

    qwen_results = json.loads(qwen_results_path.read_text(encoding="utf-8"))
    gemma_results = json.loads(gemma_results_path.read_text(encoding="utf-8"))
    q = qwen_results["principal_findings"]
    g = gemma_results["principal_findings"]
    claims = {
        "polish_improved_count": {
            "deepseek": q["deepseek_polish_improved"],
            "gemma": g["polish_improved"],
            "qwen": q["polish_improved"],
        },
        "polish_median_delta_percentage_points": {
            "deepseek": q["deepseek_polish_median_delta_percentage_points"],
            "gemma": g["polish_median_delta_percentage_points"],
            "qwen": q["polish_median_delta_percentage_points"],
        },
        "polished_above_all_fixed_seeds": {
            "deepseek": 15,
            "gemma": g["polished_above_all_fixed_seeds"],
            "qwen": q["polished_above_all_fixed_seeds"],
        },
        "focal_polished_outranks_originals": {
            "deepseek": q["paper_focal_polished_deepseek_outranks_originals"],
            "gemma": g["paper_focal_polished_gemma_outranks_originals"],
            "qwen": q["paper_focal_polished_qwen_outranks_originals"],
        },
        "local_only_median_win_rate": {
            "deepseek": q["deepseek_local_only_median_win_rate"],
            "gemma": g["local_only_median_win_rate"],
            "qwen": q["local_only_median_win_rate"],
        },
        "polished_median_win_rate": {
            "deepseek": q["deepseek_polished_median_win_rate"],
            "gemma": g["polished_median_win_rate"],
            "qwen": q["polished_median_win_rate"],
        },
        "hply_outranks_originals": {
            "deepseek": q["hply_deepseek_outranks_originals"],
            "gemma": g["hply_gemma_outranks_originals"],
            "qwen": q["hply_qwen_outranks_originals"],
        },
    }

    try:
        gemma_run_record = str(gemma_run.resolve().relative_to(HERE.parent.resolve()))
    except ValueError:
        gemma_run_record = str(gemma_run)
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qwen_run_id": qwen_run_id,
        "gemma_run": gemma_run_record,
        "correlations": correlations,
        "fundraising_correlations": fundraising_correlations,
        "claim_comparison": claims,
    }
    json_path = qwen_run / "cross-evaluator-comparison.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    correlation_rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {values['n']} | "
        f"{values['deepseek_gemma']:.3f} | {values['deepseek_qwen']:.3f} | "
        f"{values['gemma_qwen']:.3f} |"
        for name, values in correlations.items()
    )
    claim_rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {values['deepseek']:.3f} | "
        f"{values['gemma']:.3f} | {values['qwen']:.3f} |"
        for name, values in claims.items()
    )
    report = f"""# Three-evaluator comparison

This report compares the paper's DeepSeek scores with two cross-family, open-weight evaluators. All three columns refer to the same frozen concept texts; Gemma and Qwen used the same pairwise prompt and both presentation orders.

| Subset | N | DeepSeek--Gemma | DeepSeek--Qwen | Gemma--Qwen |
|---|---:|---:|---:|---:|
{correlation_rows}

Spearman correlations with realized fundraising among the 30 originals are {fundraising_correlations['deepseek']:.3f} for DeepSeek, {fundraising_correlations['gemma']:.3f} for Gemma, and {fundraising_correlations['qwen']:.3f} for Qwen. These correlations are descriptive checks on the reference set, not estimates of evaluator accuracy for generated concepts.

| Quantity | DeepSeek | Gemma | Qwen |
|---|---:|---:|---:|
{claim_rows}

Win-rate levels need not be comparable across evaluators because each judge can apply a different threshold. The most informative checks are whether rankings and the paper's ordinal conclusions survive.
"""
    report_path = qwen_run / "cross-evaluator-comparison.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-run-id", required=True)
    parser.add_argument("--gemma-run", type=Path, default=DEFAULT_GEMMA_RUN)
    arguments = parser.parse_args()
    main(arguments.qwen_run_id, arguments.gemma_run)
